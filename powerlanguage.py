"""Practical PowerLanguage interpreter for common single-instrument strategies.

This is an intentionally safe interpreter (never evals Python or accesses the OS).
It supports structured PowerLanguage blocks, multiple data streams built from the
primary OHLC input, common series functions, and the usual market/stop/limit orders.
It is not a substitute for TradeStation's proprietary runtime/compiler.
"""
from __future__ import annotations

import ast
import bisect
import datetime as _datetime
import math
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from technical_indicators import INDICATOR_ALIASES, IndicatorLibrary


class PowerLanguageError(ValueError):
    pass


def _strip_comments(source: str) -> list[tuple[int, str]]:
    rows = []
    in_block = False
    in_brace = False
    for no, raw in enumerate(source.splitlines(), 1):
        line = raw
        while True:
            if in_block:
                end = line.find("*)")
                if end < 0:
                    line = ""
                    break
                line, in_block = line[end + 2 :], False
            if in_brace:
                end = line.find("}")
                if end < 0:
                    line = ""
                    break
                line, in_brace = line[end + 1 :], False
            brace_start = line.find("{")
            start = line.find("(*")
            if brace_start >= 0 and (start < 0 or brace_start < start):
                end = line.find("}", brace_start + 1)
                if end < 0:
                    line, in_brace = line[:brace_start], True
                    break
                line = line[:brace_start] + line[end + 1 :]
                continue
            if start >= 0:
                end = line.find("*)", start + 2)
                if end < 0:
                    line, in_block = line[:start], True
                    break
                line = line[:start] + line[end + 2 :]
                continue
            break
        line = line.split("//", 1)[0].strip()
        line = re.sub(r"(?i)\bintrabarpersist\s+", "", line)
        line = re.sub(r"\{[^{}]*\}", "", line).strip()
        if re.fullmatch(r"(?i)\[\s*IntrabarOrderGeneration\s*=\s*False\s*\]", line):
            continue
        if re.fullmatch(r"(?i)\[\s*IntrabarOrderGeneration\s*=\s*True\s*\]", line):
            raise PowerLanguageError(
                f"line {no}: IntrabarOrderGeneration=True requires tick-level evaluation, which this OHLC engine does not provide"
            )
        if line:
            # Accept the common compact spelling `end else begin`.
            line = re.sub(r"(?i)\bend\s*;?\s+else\b", "End\nElse", line)
            rows.extend((no, part.strip()) for part in line.splitlines() if part.strip())
    return rows


def _value(text):
    t = text.strip().rstrip(";").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        return t[1:-1]
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    try:
        return int(t, 0)
    except ValueError:
        try:
            return float(t)
        except ValueError as exc:
            raise PowerLanguageError(f"unsupported declaration value {t!r}") from exc


def _split_declaration_parts(text: str) -> list[str]:
    parts, start, parens, brackets, quote = [], 0, 0, 0, None
    for i, char in enumerate(text):
        if quote:
            if char == quote and (i == 0 or text[i - 1] != "\\"):
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "(": parens += 1
        elif char == ")": parens -= 1
        elif char == "[": brackets += 1
        elif char == "]": brackets -= 1
        elif char == "," and parens == 0 and brackets == 0:
            parts.append(text[start:i].strip()); start = i + 1
    if text[start:].strip(): parts.append(text[start:].strip())
    return parts


def _split_statements(text: str) -> list[str]:
    """Split PowerLanguage statements at semicolons outside string literals."""
    parts, start, quote = [], 0, None
    for i, char in enumerate(text):
        if quote:
            if char == quote and (i == 0 or text[i - 1] != "\\"):
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == ";":
            if text[start:i].strip():
                parts.append(text[start:i].strip())
            start = i + 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def parse_source(source: str):
    rows = _strip_comments(source)
    # PowerLanguage diagnostic Print calls do not affect strategy decisions.
    # Replace balanced calls (including multiline/formatted arguments) with a
    # no-op statement so surrounding `if ... then Print(...)` remains valid.
    cleaned, in_print, depth, print_line = [], False, 0, 0
    for no, raw in rows:
        line = raw
        # Chart/study attributes and DLL imports are platform declarations. In
        # the historical engine their data functions are supplied natively and
        # chart settings are handled by the runner.
        iog_true = re.match(r"(?i)^\s*\[\s*IntrabarOrderGeneration\s*=\s*True\s*\]\s*$", line)
        if iog_true:
            raise PowerLanguageError(f"line {no}: IntrabarOrderGeneration=True needs tick-level fills and cannot be reproduced with OHLC bars")
        if re.match(r"(?i)^\s*\[[^\]]+\]\s*$", line):
            continue
        if re.match(r"(?i)^\s*DefineDLLFunc\s*:", line):
            continue
        if not in_print:
            m = re.search(r"(?i)\bPrint\s*\(", line)
            if m:
                prefix = line[:m.start()].rstrip()
                if prefix:
                    cleaned.append((no, prefix))
                in_print, depth, print_line = True, 1, no
                tail = line[m.end():]
            else:
                cleaned.append((no, line)); continue
        else:
            tail = line
        depth += tail.count("(") - tail.count(")")
        if depth <= 0:
            cleaned.append((print_line, "NoOp"))
            in_print = False
    rows = cleaned
    # Join wrapped conditions and expressions, which are common in EL files.
    joined = []
    k = 0
    while k < len(rows):
        no, line = rows[k]
        if re.match(r"(?i)^\s*(?:BuyToCover|Sell\s+Short|SellShort|Buy|Sell)\b", line):
            parts = [line]; k += 1
            while k < len(rows) and not re.search(r"(?i)\b(?:next\s+bar|this\s+bar)\b", " ".join(parts)):
                parts.append(rows[k][1]); k += 1
            joined.append((no, " ".join(parts))); continue
        if re.match(r"(?i)^if\s+", line) and not re.search(r"(?i)\bthen\b", line):
            parts = [line]; k += 1
            while k < len(rows) and not re.search(r"(?i)\bthen\b", " ".join(parts)):
                parts.append(rows[k][1]); k += 1
            joined.append((no, " ".join(parts))); continue
        continuation_start = k + 1 < len(rows) and re.match(r"(?i)^\s*(?:and|or)\b", rows[k + 1][1])
        if continuation_start or line.count("(") > line.count(")") or re.search(r"(?i)(?:=|[+*/-]|\b(?:and|or))\s*$", line):
            parts = [line]; k += 1
            while k < len(rows) and (sum(p.count("(")-p.count(")") for p in parts) > 0 or
                                     re.search(r"(?i)(?:=|[+*/-]|\b(?:and|or))\s*$", parts[-1]) or
                                     re.match(r"(?i)^\s*(?:and|or)\b", rows[k][1])):
                parts.append(rows[k][1]); k += 1
            joined.append((no, " ".join(parts))); continue
        joined.append((no, line)); k += 1
    rows = joined
    inputs, variables, code, section = {}, {}, [], None
    for no, line in rows:
        head = re.match(r"^(Input|Inputs|Var|Vars|Variable|Variables|Array|Arrays)\s*:\s*(.*)$", line, re.I)
        if head:
            key = head.group(1).lower()
            section = "inputs" if key.startswith("input") else ("arrays" if key.startswith("array") else "variables")
            line = head.group(2).strip()
            if not line:
                continue
        if section:
            # Declaration continuation ends at the first executable statement.
            parts = _split_declaration_parts(line.rstrip(";"))
            if section == "arrays":
                parsed_arrays = []
                for part in parts:
                    am = re.fullmatch(r"(?:IntraBarPersist\s+)?([A-Za-z_]\w*)\s*\[\s*([^\]]*)\s*\]\s*\(\s*(.*?)\s*\)", part, re.I)
                    if not am:
                        parsed_arrays = []
                        break
                    name, dimension_text, initial_text = am.groups()
                    dims = [] if not dimension_text.strip() else [int(_value(x.strip())) + 1 for x in dimension_text.split(",")]
                    initial = _value(initial_text)
                    parsed_arrays.append((name.lower(), dims, initial))
                if parsed_arrays:
                    def make_array(dimensions, initial):
                        if not dimensions:
                            return [initial]
                        if len(dimensions) == 1:
                            return [initial for _ in range(dimensions[0])]
                        return [make_array(dimensions[1:], initial) for _ in range(dimensions[0])]
                    for name, dims, initial in parsed_arrays:
                        variables[name] = make_array(dims, initial)
                    continue
            decls = []
            for part in parts:
                m = re.fullmatch(r"([A-Za-z_]\w*)\s*(?:\(\s*(.*?)\s*\)|=\s*(.*))", part)
                if not m:
                    decls = []
                    break
                name = m.group(1).lower()
                val = m.group(2) if m.group(2) is not None else m.group(3)
                try:
                    if section == "inputs" and re.fullmatch(r"(?i)(open|high|low|close|volume)", val.strip()):
                        decls.append((name, ("__series_default__", val.strip().lower())))
                    else:
                        decls.append((name, _value(val)))
                except PowerLanguageError:
                    decls = []
                    break
            if decls:
                target = inputs if section == "inputs" else variables
                target.update(decls)
                continue
            section = None
        # PowerLanguage uses semicolons as statement terminators, and scripts
        # commonly put several simple assignments on one physical line.
        parts = _split_statements(line)
        part_index = 0
        while part_index < len(parts):
            part = parts[part_index]
            inline_for = re.match(
                r"(?is)^(for\s+\w+\s*=.+?\s+(?:to|downto)\s+.+?\s+begin)\s+(.+)$",
                part,
            )
            if (inline_for and part_index + 1 < len(parts)
                    and parts[part_index + 1].strip().lower() == "end"):
                code.extend(((no, inline_for.group(1)), (no, inline_for.group(2)), (no, "end")))
                part_index += 2
            else:
                code.append((no, part))
                part_index += 1

    def one(no, text, i):
        text = text.strip().rstrip(";")
        if text.lower().startswith("begin"):
            return parse_seq(i, True)
        if text.lower().startswith("if "):
            return parse_if(no, text, i)
        if re.match(r"(?i)^once\s+begin\b", text):
            body, end_i = parse_seq(i + 1, True)
            return [(no, "once", body)], end_i
        fm = re.match(r"(?is)^for\s+(\w+)\s*=\s*(.+?)\s+(to|downto)\s+(.+?)\s+begin\s*$", text)
        if fm:
            body, end_i = parse_seq(i, True)
            return [(no, "for", fm.group(1).lower(), fm.group(2), fm.group(3).lower(), fm.group(4), body)], end_i
        wm = re.match(r"(?is)^while\s+(.+?)\s+begin\s*$", text)
        if wm:
            body, end_i = parse_seq(i, True)
            return [(no, "while", wm.group(1), body)], end_i
        return [(no, "statement", text)], i

    def parse_if(no, text, i):
        m = re.match(r"(?is)^if\s+(.+?)\s+then\s*(.*)$", text.rstrip(";"))
        if not m:
            raise PowerLanguageError(f"line {no}: expected If ... Then")
        condition, rest = m.group(1).strip(), m.group(2).strip()
        inline_else = re.search(r"(?i)\s+else\s+", rest)
        if inline_else and not rest.lower().startswith("begin"):
            yes_text, no_text = rest[:inline_else.start()].strip(), rest[inline_else.end():].strip()
            yes, i = one(no, yes_text, i)
            no_branch, i = one(no, no_text, i)
            return [(no, "if", condition, yes, no_branch)], i
        if rest:
            if rest.lower().startswith("begin"):
                yes, i = parse_seq(i, True)
            else:
                yes, i = one(no, rest, i)
        elif i < len(code) and code[i][1].strip().lower().rstrip(";") == "begin":
            yes, i = parse_seq(i + 1, True)
        elif i < len(code):
            yes, i = one(code[i][0], code[i][1], i + 1)
        else:
            raise PowerLanguageError(f"line {no}: If has no statement after Then")
        no_branch = []
        if i < len(code) and code[i][1].strip().lower().startswith("else"):
            eno, es = code[i]
            rest = re.sub(r"(?i)^else\s*", "", es.strip()).strip()
            i += 1
            if rest:
                no_branch, i = one(eno, rest, i)
            elif i < len(code):
                no_branch, i = one(code[i][0], code[i][1], i + 1)
        return [(no, "if", condition, yes, no_branch)], i

    def parse_seq(i, stop_end=False):
        seq = []
        while i < len(code):
            no, raw = code[i]
            s = raw.strip()
            low = s.lower().rstrip(";").strip()
            if low == "end":
                if not stop_end:
                    raise PowerLanguageError(f"line {no}: unexpected End")
                return seq, i + 1
            if low.startswith("else"):
                return seq, i
            if low == "once" and i + 1 < len(code) and code[i + 1][1].strip().lower().rstrip(";").strip() == "begin":
                # MultiCharts commonly puts `once` and `begin` on separate
                # lines. Treat it exactly like `once begin`.
                body, i = parse_seq(i + 2, True)
                seq.append((no, "once", body))
            elif low.startswith("begin"):
                nested, i = parse_seq(i + 1, True)
                seq.append((no, "block", nested))
            elif low.startswith("if "):
                node, i = parse_if(no, s, i + 1)
                seq.extend(node)
            elif re.match(r"(?i)^once\s+begin\b", s):
                body, i = parse_seq(i + 1, True)
                seq.append((no, "once", body))
            elif re.match(r"(?i)^for\s+\w+\s*=.+\s+(?:to|downto)\s+.+\s+begin\s*$", s):
                node, i = one(no, s, i + 1); seq.extend(node)
            elif re.match(r"(?i)^while\s+.+\s+begin\s*$", s):
                node, i = one(no, s, i + 1); seq.extend(node)
            elif low in ("", ";"):
                i += 1
            else:
                seq.append((no, "statement", s.rstrip(";")))
                i += 1
        if stop_end:
            raise PowerLanguageError("unterminated Begin block")
        return seq, i

    tree, _ = parse_seq(0)
    return inputs, variables, tree


def _safe_expression(text: str, declared_names=None) -> str:
    """Normalize and validate an expression before embedding it in compiled code."""
    normalized = _normalize_expr(text, declared_names)
    try:
        node = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise PowerLanguageError(f"invalid PowerLanguage expression {text!r}: {exc}") from exc
    class LazyIff(ast.NodeTransformer):
        def visit_Call(self, call):
            call = self.generic_visit(call)
            if isinstance(call.func, ast.Name) and call.func.id.lower() in {"iff", "iffstring"}:
                if len(call.args) != 3 or call.keywords:
                    raise PowerLanguageError("IFF/IFFString expects exactly three positional arguments")
                return ast.copy_location(ast.IfExp(test=call.args[0], body=call.args[1], orelse=call.args[2]), call)
            return call
    node = LazyIff().visit(node)
    node = ast.fix_missing_locations(node)
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, ast.BoolOp,
               ast.Compare, ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
               ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
               ast.Gt, ast.GtE, ast.FloorDiv, ast.Tuple, ast.IfExp)
    if any(not isinstance(item, allowed) for item in ast.walk(node)):
        raise PowerLanguageError(f"unsupported expression syntax: {text}")
    return ast.unparse(node.body)


def compile_program(tree, declared_names=None):
    """Compile the parsed statement tree once; the result is reused per bar."""
    output = []
    def safe_expr(text):
        return _safe_expression(text, declared_names)

    def emit(nodes, indent=0):
        pad = "    " * indent
        for node in nodes:
            line, kind = node[0], node[1]
            if kind == "block":
                emit(node[2], indent)
                continue
            if kind == "once":
                output.append(f"{pad}if not __once_done:")
                output.append(f"{pad}    __once_done = True")
                if node[2]: emit(node[2], indent + 1)
                continue
            if kind == "for":
                var, start, direction, end, body = node[2], safe_expr(node[3]), node[4], safe_expr(node[5]), node[6]
                if direction == "downto":
                    output.append(f"{pad}for {var} in range(int({start}), int({end}) - 1, -1):")
                else:
                    output.append(f"{pad}for {var} in range(int({start}), int({end}) + 1):")
                if body: emit(body, indent + 1)
                else: output.append(f"{pad}    pass")
                continue
            if kind == "while":
                cond = safe_expr(node[2])
                counter = f"__pl_while_count_{line}"
                output.append(f"{pad}{counter} = 0")
                output.append(f"{pad}while {cond}:")
                output.append(f"{pad}    {counter} += 1")
                output.append(f"{pad}    if {counter} > 100000:")
                output.append(f"{pad}        __while_limit__({line})")
                if node[3]: emit(node[3], indent + 1)
                else: output.append(f"{pad}    pass")
                continue
            if kind == "if":
                cond = safe_expr(node[2])
                output.append(f"{pad}if {cond}:")
                if node[3]: emit(node[3], indent + 1)
                else: output.append(f"{pad}    pass")
                if node[4]:
                    output.append(f"{pad}else:")
                    emit(node[4], indent + 1)
                continue

            stmt = node[2].strip()
            if stmt.lower() in ("noop", "nop"):
                output.append(f"{pad}pass")
                continue
            call = re.match(r"^(SetStopLoss|SetProfitTarget)(_pt)?\s*\((.*)\)$", stmt, re.I)
            if call:
                kind_name = "loss" if call.group(1).lower() == "setstoploss" else "target"
                helper = "__set_stop_pt__" if call.group(2) else "__set_stop__"
                output.append(f"{pad}{helper}({kind_name!r}, {safe_expr(call.group(3))}, {line})")
                continue
            if re.match(r"^SetStop(Contract|Position|Share)\s*(?:\(\s*\))?$", stmt, re.I):
                per_contract = bool(re.match(r"(?i)^SetStop(Contract|Share)", stmt))
                output.append(f"{pad}__set_stop_mode__({per_contract!r})")
                continue
            if re.match(r"(?i)^SetExitOnClose\s*(?:\(\s*\))?$", stmt):
                output.append(f"{pad}__set_exit_on_close__({line})")
                continue
            plot = re.match(r"(?i)^Plot(\d*)\s*\((.*)\)$", stmt)
            if plot:
                plot_number = int(plot.group(1) or 1)
                args = _split_declaration_parts(plot.group(2))
                if not args:
                    raise PowerLanguageError(f"line {line}: Plot expects a value")
                output.append(f"{pad}__plot__({plot_number}, {safe_expr(args[0])}, {line})")
                continue
            no_plot = re.match(r"(?i)^NoPlot\s*\(\s*(\d+)\s*\)$", stmt)
            if no_plot:
                output.append(f"{pad}__no_plot__({int(no_plot.group(1))}, {line})")
                continue
            runtime_error = re.match(r"(?i)^RaiseRunTimeError\s*\((.*)\)$", stmt)
            if runtime_error:
                output.append(f"{pad}__raise_runtime_error__({safe_expr(runtime_error.group(1))}, {line})")
                continue
            risk_call = re.match(r"(?i)^(SetBreakEven|SetDollarTrailing|SetPercentTrailing|SetTrailingStop)(_pt)?\s*\((.*)\)$", stmt)
            if risk_call:
                if risk_call.group(1).lower() == "settrailingstop" and not risk_call.group(2):
                    raise PowerLanguageError(f"line {line}: MultiCharts supports SetTrailingStop_pt, not SetTrailingStop")
                args = _split_declaration_parts(risk_call.group(3))
                encoded = ", ".join(safe_expr(arg) for arg in args)
                risk_kind = risk_call.group(1).lower() + ("_pt" if risk_call.group(2) else "")
                output.append(f"{pad}__set_risk_order__({risk_kind!r}, ({encoded}), {line})")
                continue
            if re.match(r"(?i)^(?:PlotPaintBar|Text_New|Text_SetString|Text_SetColor|Text_SetStyle)\s*\(.*\)$", stmt):
                output.append(f"{pad}pass")
                continue
            if re.match(r"(?i)^(?:(?:TL|Text|Arw|Rectangle|Ellipse|TrendLine|VerticalLine|HorizontalLine)_\w*|Alert|SetPlotColor|SetPlotWidth|SetPlotBGColor|SetPlotType|SetPlotStyle)\s*\(", stmt):
                output.append(f"{pad}pass")
                continue
            order = re.match(r"^(BuyToCover|Sell\s+Short|SellShort|Buy|Sell)\s*(.*)$", stmt, re.I)
            if order:
                action, tail = re.sub(r"\s+", "", order.group(1)).lower(), order.group(2)
                unit = r"(?:contracts?|shares?|lots?|units?)"
                numeric_qty = r"((?:\d+(?:\.\d*)?|\.\d+))"
                all_qty = re.search(rf"\bAll\s+{unit}\b", tail, re.I)
                qm = re.search(rf"\)\s*(\w+)\s+{unit}\b", tail, re.I) or re.search(rf"\b{numeric_qty}\s+{unit}\b", tail, re.I)
                qty_expr = "currentcontracts" if all_qty else (safe_expr(qm.group(1)) if qm else "1")
                label_match = re.match(r"\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", tail)
                order_label = label_match.group(1) if label_match else ""
                price_match = re.search(r"\bat\s+(.+?)\s+(Limit|Stop)\b", tail, re.I)
                mode = price_match.group(2).lower() if price_match else "market"
                price_expr = safe_expr(price_match.group(1)) if price_match else "None"
                this_close = bool(re.search(r"\bthis\s+bar\s+(?:on\s+)?close\b", tail, re.I))
                this_open = bool(re.search(r"\bthis\s+bar\s+(?:on\s+)?open\b", tail, re.I))
                output.append(f"{pad}__order__({action!r}, {qty_expr}, {mode!r}, {price_expr}, {this_close!r}, {this_open!r}, {line}, {order_label!r})")
                continue
            assign = re.match(r"^(\w+)\s*=\s*(.+)$", stmt)
            if assign:
                name, expr = assign.group(1).lower(), safe_expr(assign.group(2))
                output.append(f"{pad}{name} = {expr}")
                continue
            if not stmt:
                continue
            array_assign = re.match(r"(?is)^([A-Za-z_]\w*)\s*\[\s*(.*?)\s*\]\s*=\s*(.+)$", stmt)
            if array_assign:
                arr_name, indices, value = array_assign.groups()
                # Direct list assignment is materially faster for nested
                # PowerLanguage profile loops; expression validation above
                # still prevents arbitrary Python syntax from entering here.
                target = arr_name.lower() + "".join(
                    f"[int({safe_expr(index.strip())})]" for index in _split_declaration_parts(indices)
                )
                output.append(f"{pad}{target} = {safe_expr(value)}")
                continue
            if re.fullmatch(r"(?i)break|continue", stmt):
                output.append(f"{pad}{stmt.lower()}")
                continue
            # Only accept array statements with an implemented runtime semantic.
            # Do not compile a call and let it fail later as a Python NameError.
            array_call = re.match(r"(?i)^(Array_SetMaxIndex|Array_SetValRange|Array_SetFloatValue|Array_SetIntegerValue|Array_SetBooleanValue|Array_SetStringValue|Array_Copy|Array_Sort|Fill_Array)\s*\(.*\)$", stmt)
            if array_call:
                output.append(f"{pad}{safe_expr(stmt)}")
                continue
            raise PowerLanguageError(f"line {line}: unsupported statement {stmt!r}; external I/O, DLL, portfolio, live DOM and chart UI effects are unavailable in historical bar simulation")

    emit(tree)
    source = "\n".join(output) or "pass"
    try:
        return compile(source, "<PowerLanguage strategy>", "exec")
    except SyntaxError as exc:
        raise PowerLanguageError(f"could not compile strategy: {exc}") from exc


def _calc_time(hhmm, minute_offset):
    """PowerLanguage CalcTime: add minutes to an HHMM time and wrap midnight."""
    try:
        value = int(float(hhmm))
        offset = int(float(minute_offset))
    except (TypeError, ValueError, OverflowError) as exc:
        raise PowerLanguageError("CalcTime expects a valid HHMM time and a numeric minute offset") from exc
    hour, minute = divmod(value, 100)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise PowerLanguageError(f"CalcTime received an invalid HHMM time: {value}")
    total = (hour * 60 + minute + offset) % (24 * 60)
    return (total // 60) * 100 + total % 60


def _normalize_expr(expr: str, declared_names=None) -> str:
    expr = expr.strip().rstrip(";")
    # Preserve literal text while normalizing case-sensitive reserved words.
    literals = []
    def protect_literal(match):
        literals.append(match.group(0))
        return f"__pl_string_{len(literals) - 1}__"
    expr = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", protect_literal, expr)
    # PowerLanguage commonly writes HHMM literals with a leading zero (0930).
    # Python rejects those integer literals, while their intended value is decimal 930.
    expr = re.sub(r"(?<![\w.])0+(\d+)(?![\w.])", r"\1", expr)
    # Explicit historical references to indicators on the secondary data
    # stream. The current Data2 bar is treated as in progress, so [1] is the
    # most recently completed Data2 bar when evaluated on Data1.
    expr = re.sub(r"\b(XAverage|Average|StdDev)\s*\(\s*(Open|High|Low|Close|Volume)\s*,\s*([^()]*)\s*\)\s*\[\s*([^\]]+)\s*\]\s+of\s+Data2",
                  lambda m: f"{m.group(1).lower()}_data2_at({m.group(2).lower()}, {m.group(3)}, {m.group(4)})", expr, flags=re.I)
    expr = re.sub(r"\bAvgTrueRange\s*\(\s*([^()]*)\s*\)\s*\[\s*([^\]]+)\s*\]\s+of\s+Data2",
                  r"avgtruerange_data2(\1, \2)", expr, flags=re.I)
    expr = re.sub(r"\bCurrentBar\s+of\s+Data2\b", "currentbar_data2", expr, flags=re.I)
    expr = re.sub(
        r"\b(Average|XAverage|StdDev|Highest|Lowest|Summation)\s*\(\s*(Open|High|Low|Close|Volume)\s+of\s+Data2\s*,\s*([^()]*)\s*\)",
        lambda m: f"{m.group(1).lower()}_data2({m.group(2).lower()}_data2, {m.group(3)})",
        expr, flags=re.I,
    )
    expr = re.sub(r"\bMarketPosition\s*\[\s*(\d+)\s*\]", r"marketposition_at(\1)", expr, flags=re.I)
    # Brackets mean bar history; parentheses select the current/prior trade.
    expr = re.sub(r"\bMarketPosition\s*\(\s*(\d+)\s*\)", r"marketposition_trade(\1)", expr, flags=re.I)
    for pl_name, runtime_name in (("EntryPrice", "closed_entry_price"), ("ExitPrice", "closed_exit_price"),
                                  ("EntryDate", "closed_entry_date"), ("EntryTime", "closed_entry_time"),
                                  ("ExitDate", "closed_exit_date"), ("ExitTime", "closed_exit_time")):
        expr = re.sub(rf"\b{pl_name}\s*\(\s*1\s*\)", f"{runtime_name}()", expr, flags=re.I)
    expr = re.sub(r"\bDate\s+of\s+Data2\b", "date_data2", expr, flags=re.I)
    def normalize_stddev(match):
        value, length, data_type = match.groups()
        value_name = value.strip().lower()
        if value_name in {"open", "high", "low", "close", "volume"}:
            return f"stddev({value}, {length}, {data_type})"
        if re.fullmatch(r"[A-Za-z_]\w*", value_name):
            return f"stddev_variable('{value_name}', {value}, {length}, {data_type})"
        return match.group(0)
    expr = re.sub(r"\b(?:StandardDev|StdDev|StDev)\s*\(\s*([^,()]+)\s*,\s*([^,()]+)\s*,\s*([^()]+)\s*\)",
                  normalize_stddev, expr, flags=re.I)
    def normalize_stddev_two(match):
        value, length = match.groups()
        value_name = value.strip().lower()
        if value_name in {"open", "high", "low", "close", "volume"}:
            return f"stddev({value}, {length})"
        if re.fullmatch(r"[A-Za-z_]\w*", value_name):
            return f"stddev_variable('{value_name}', {value}, {length})"
        return match.group(0)
    expr = re.sub(r"\b(?:StandardDev|StdDev|StDev)\s*\(\s*([^,()]+)\s*,\s*([^()]+)\s*\)",
                  normalize_stddev_two, expr, flags=re.I)
    expr = re.sub(r"\bADX\s*\(\s*([^()]*)\s*\)\s*\[\s*(\d+)\s*\]",
                  r"adx_at(\1, \2)", expr, flags=re.I)
    expr = re.sub(r"\b(Open|High|Low|Close|Volume)\s*\[\s*(\d+)\s*\]\s+of\s+Data2\b", lambda m: f"{m.group(1).lower()}_data2_at({m.group(2)})", expr, flags=re.I)
    expr = re.sub(r"\b(Open|High|Low|Close|Volume)\s*\[\s*(\d+)\s*\]\s+of\s+Data1\b", lambda m: f"{m.group(1).lower()}_at({m.group(2)})", expr, flags=re.I)
    expr = re.sub(r"\b(Open|High|Low|Close|Volume)\s+of\s+Data2\s*\[\s*(\d+)\s*\]", lambda m: f"{m.group(1).lower()}_data2_at({m.group(2)})", expr, flags=re.I)
    # Common EasyLanguage crossing operator (simple names and one-level calls).
    operand = r"(?:[A-Za-z_]\w*\s+of\s+Data\d+|[A-Za-z_]\w*(?:\([^()]*\))?)"
    cross_pattern = re.compile(
        rf"({operand})\s+crosses\s+(over|above|under|below)\s+({operand})",
        re.I,
    )
    while True:
        match = cross_pattern.search(expr)
        if not match:
            break
        op = "over" if match.group(2).lower() in ("over", "above") else "under"
        key = f"{match.group(1).lower()}_{op}_{match.group(3).lower()}"
        replacement = f"crosses_{op}(({match.group(1)}),({match.group(3)}),'{key}')"
        expr = expr[:match.start()] + replacement + expr[match.end():]
    expr = re.sub(r"\bEntryPrice\s*\(\s*0\s*\)", "entryprice", expr, flags=re.I)
    expr = re.sub(r"\bBarNumber\b", "currentbar", expr, flags=re.I)
    expr = re.sub(r"\bBarsSinceEntry\s*\(\s*0\s*\)", "barssinceentry", expr, flags=re.I)
    expr = re.sub(r"\bTime\s+of\s+Data2\b", "time_data2", expr, flags=re.I)
    expr = re.sub(r"\b(Date|Time)\b", lambda m: m.group(1).lower(), expr, flags=re.I)
    expr = _rewrite_mod_calls(expr)
    expr = re.sub(r"\b(Open|High|Low|Close|Volume)\s+of\s+Data2\b", lambda m: m.group(1).lower() + "_data2", expr, flags=re.I)
    expr = re.sub(r"\bAvgTrueRange\s*\(\s*(.*?)\s*\)\s*\[\s*(\d+)\s*\]", r"avgtruerange(\1,\2)", expr, flags=re.I)
    def normalize_rolling_history(match):
        kind, source, length, ago = match.groups()
        kind = kind.lower()
        source_name = source.strip().lower()
        if source_name in {"open", "high", "low", "close", "volume"}:
            return f"{kind}_at({source}, {length}, {ago})"
        return f"rolling_variable_at('{source_name}', {source}, '{kind}', {length}, {ago})"
    expr = re.sub(r"\b(Highest|Lowest|Average|XAverage)\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^()]+?)\s*\)\s*\[\s*(\d+)\s*\]",
                  normalize_rolling_history, expr, flags=re.I)
    def normalize_rolling_current(match):
        kind, source, length = match.groups()
        source_name = source.strip().lower()
        length_expr = length.strip()
        # Dynamic lookbacks (for example Highest(High, BarsSinceEntry + 1))
        # must be evaluated only for the current bar. Routing each distinct
        # lookback through rolling() would cache a full-length array per value
        # and grow toward quadratic work/memory during long trades.
        if (kind.lower() in {"highest", "highestfc", "lowest", "lowestfc"}
                and source_name in {"open", "high", "low", "close", "volume"}
                and not re.fullmatch(r"\d+", length_expr)):
            return f"rolling_dynamic('{kind.lower()}', {source}, {length_expr})"
        if source_name in {"open", "high", "low", "close", "volume"}:
            return match.group(0)
        return f"rolling_variable('{source_name}', {source}, '{kind.lower()}', {length})"
    expr = re.sub(r"\b(Highest|Lowest|Average|XAverage)\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^()]+?)\s*\)",
                  normalize_rolling_current, expr, flags=re.I)
    expr = re.sub(r"\bTrueRange\s*\(\s*\)", "truerange()", expr, flags=re.I)
    # A few built-in studies are used as zero-argument series in legacy
    # PowerLanguage code (rather than written with an explicit call).
    zero_argument_studies = (
        "OBV|OnBalanceVolume|VWAP|TypicalPrice|WeightedClose|MedianPrice|AwesomeOscillator|"
        "AcceleratorOscillator|AccumDist|AccumulationDistribution|PVT|Doji|InsideBar|OutsideBar|"
        "BullEngulfing|BearEngulfing|BullHarami|BearHarami|Hammer|HangingMan|ShootingStar|MorningStar|EveningStar|"
        "MorningDoji|EveningDoji|PiercingLine|DarkCloudCover|ThreeWhiteSoldiers|ThreeBlackCrows|"
        "KeyReversalUp|KeyReversalDown|GapUp|GapDown|"
        "VolumeToday|FirstBarOfDay|FirstBarOfWeek|FirstBarOfMonth|FirstBarOfYear|DayOpen|DayHigh|DayLow|"
        "BidAskSpread|BidAskMidpoint|QuotePrice|BidAskVolRatio|BidAskVolumeRatio|BidAndAskVolRatio|"
        "NewHighForChart|NewLowForChart|NewHighForDay|NewLowForDay|NewHighForWeek|NewLowForWeek|"
        "NewHighForMonth|NewLowForMonth|NewHighForYear|NewLowForYear|AverageCloseLessThanAverageOpen|"
        "AverageCloseGreaterThanAverageOpen|AvgCloseLessThanAvgOpen|AvgCloseGreaterThanAvgOpen|"
        "NextThirdFriday|Expiration|ConsecutiveUps|ConsecutiveDowns|MarketDepthImbalance|MarketDepthSpread|MarketDepthMicroprice|"
        "AccumSwingIndex|AccumulativeSwingIndex|SwingIndex|MomentumIncreasing|MomentumDecreasing|IslandReversalUp|IslandReversalDown|"
        "OFB_HistCount|OFB_DepthCount|OFB_GexCount|OFB_CotCount|OFB_Version|OFB_Status|OFB_AgeMs|"
        "OFB_DepthLevels|OFB_GexStrikes|OFB_GexCentroid|OFB_GexMajPos|OFB_GexMajNeg|OFB_GexCentroidDay|"
        "OFB_GexPosDay|OFB_GexNegDay|OFB_GexPosOI|OFB_GexNegOI|OFB_GexBasis|OFB_GexDTE|"
        "OFB_GexFlip|OFB_GexFlipDay|OFB_GexFlipOI|OFB_GexSpot|OFB_GexFlipSteep|"
        "OFB_GexFlipSteepDay|OFB_GexFlipSteepOI|"
        "AlligatorJaw|AlligatorTeeth|AlligatorLips|ChaosAlligatorJaw|ChaosAlligatorTeeth|ChaosAlligatorLips|"
        "GatorUpper|GatorLower|ChaosGatorUpper|ChaosGatorLower|ChaosFractalHigh|ChaosFractalLow|MarketDepthBBOImbalance|MarketDepthTopImbalance|"
        "NewIntradayHigh|NewIntradayLow|NewIntradayHiLo|NewNDayHigh|NewNDayLow|MarketDepthBBOImbalance|MarketDepthTopImbalance|"
        "MarketDepthWeightedMid|MarketDepthWeightedMidPrice|ChaosFractalHigh|ChaosFractalLow|MarneyVolume|MarneyVolumeIndicator|MarneyRange|MarneyRangeIndicator"
        "|MarketDepthBid|MarketDepthAsk|MarketDepthBidPrice|MarketDepthAskPrice|MarketDepthBidSize|MarketDepthAskSize|"
        "FloorTraderPivot|FloorTraderPivots|PivotPoint|FloorTraderResistance1|FloorTraderSupport1|YesterdayHigh|YesterdayLow|YesterdayClose|"
        "ZigZagPercent|ZigZagPoints|AccumDistBuyPr|AccumDistPrVol|MoneyFlow|Volatility|"
        "PriceGreaterThanAverage|PriceLessThanAverage"
    )
    declared_names = {str(name).lower() for name in (declared_names or ())}
    study_names = [name for name in zero_argument_studies.split("|") if name.lower() not in declared_names]
    study_pattern = "|".join(re.escape(name) for name in study_names)
    expr = re.sub(rf"\b({study_pattern})\b(?!\s*\()", r"\1()", expr, flags=re.I)
    expr = re.sub(r"\b(Close|Open|High|Low|Volume|Bid|Ask|BidSize|AskSize|TradeCount)\s*\[\s*([^\]]+)\s*\]", lambda m: f"{m.group(1).lower()}_at({m.group(2)})", expr, flags=re.I)
    expr = re.sub(r"\b(Date|Time)\s*\[\s*([^\]]+)\s*\]", lambda m: f"{m.group(1).lower()}_at({m.group(2)})", expr, flags=re.I)
    def normalize_variable_index(match):
        indices = tuple(x.strip() for x in match.group(2).split(","))
        index_expr = "(" + ",".join(indices) + ("," if len(indices) == 1 else "") + ")"
        return f"series_at('{match.group(1).lower()}', {index_expr})"
    expr = re.sub(r"\b([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]", normalize_variable_index, expr)
    expr = re.sub(r"\bBarStatus\s*\(\s*1\s*\)", "barstatus1", expr, flags=re.I)
    expr = re.sub(r"\bBarStatus\s*\(\s*2\s*\)", "barstatus2", expr, flags=re.I)
    expr = expr.replace("<>", "!=")
    expr = re.sub(r"\bmod\b", "%", expr, flags=re.I)
    expr = re.sub(r"\bdiv\b", "//", expr, flags=re.I)
    expr = expr.replace("^", "**")
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    expr = re.sub(r"\btrue\b", "True", expr, flags=re.I)
    expr = re.sub(r"\bfalse\b", "False", expr, flags=re.I)
    expr = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: m.group(0).lower(), expr)
    expr = re.sub(r"\btrue\b", "True", expr, flags=re.I)
    expr = re.sub(r"\bfalse\b", "False", expr, flags=re.I)
    expr = _rewrite_derived_rolling_calls(expr)
    expr = _rewrite_swing_calls(expr)
    expr = _rewrite_ofb_reference_calls(expr)
    for index, literal in enumerate(literals):
        expr = expr.replace(f"__pl_string_{index}__", literal)
    return expr


def _rewrite_mod_calls(expr: str) -> str:
    """Rewrite Mod(a, b) safely, including nested calls and array indices."""
    pattern = re.compile(r"\bmod\s*\(", re.I)
    search_from = 0
    while True:
        match = pattern.search(expr, search_from)
        if not match:
            return expr
        open_paren = expr.find("(", match.start())
        depth, close_paren = 0, None
        for index in range(open_paren, len(expr)):
            if expr[index] == "(":
                depth += 1
            elif expr[index] == ")":
                depth -= 1
                if depth == 0:
                    close_paren = index
                    break
        if close_paren is None:
            return expr
        args = _split_declaration_parts(expr[open_paren + 1:close_paren])
        if len(args) != 2:
            search_from = close_paren + 1
            continue
        replacement = f"(({args[0]}) % ({args[1]}))"
        expr = expr[:match.start()] + replacement + expr[close_paren + 1:]
        search_from = match.start() + len(replacement)


def _rewrite_derived_rolling_calls(expr: str) -> str:
    """Route rolling indicators on computed expressions through per-bar history."""
    function_map = {
        "summation": "sum", "average": "average", "averagefc": "average",
        "xaverage": "xaverage", "highest": "highest", "highestfc": "highest",
        "lowest": "lowest", "stddev": "stddev", "standarddev": "stddev", "stdev": "stddev",
    }
    search_from = 0
    pattern = re.compile(r"\b(" + "|".join(function_map) + r")\s*\(", re.I)
    while True:
        match = pattern.search(expr, search_from)
        if not match:
            break
        open_paren = expr.find("(", match.start())
        depth = 0
        close_paren = None
        for index in range(open_paren, len(expr)):
            if expr[index] == "(":
                depth += 1
            elif expr[index] == ")":
                depth -= 1
                if depth == 0:
                    close_paren = index
                    break
        if close_paren is None:
            break
        arguments = _split_declaration_parts(expr[open_paren + 1:close_paren])
        if len(arguments) < 2:
            search_from = close_paren + 1
            continue
        source = arguments[0].strip()
        if re.fullmatch(r"(?:open|high|low|close|volume)", source, re.I):
            search_from = close_paren + 1
            continue
        kind = function_map[match.group(1).lower()]
        data_type = arguments[2].strip() if len(arguments) > 2 else "1"
        replacement = (
            f"rolling_expression({kind!r}, {source!r}, ({source}), "
            f"{arguments[1].strip()}, {data_type})"
        )
        expr = expr[:match.start()] + replacement + expr[close_paren + 1:]
        search_from = match.start() + len(replacement)
    return expr


def _rewrite_swing_calls(expr: str) -> str:
    """Map MultiCharts SwingHigh/SwingLow calls to causal, bar-indexed helpers."""
    pattern = re.compile(r"\b(swinghigh|swinglow)\s*\(", re.I)
    search_from = 0
    while True:
        match = pattern.search(expr, search_from)
        if not match:
            return expr
        open_paren = expr.find("(", match.start())
        depth, close_paren = 0, None
        for index in range(open_paren, len(expr)):
            if expr[index] == "(":
                depth += 1
            elif expr[index] == ")":
                depth -= 1
                if depth == 0:
                    close_paren = index
                    break
        if close_paren is None:
            return expr
        args = _split_declaration_parts(expr[open_paren + 1:close_paren])
        if len(args) != 4:
            raise PowerLanguageError(f"{match.group(1)} expects Instance, Price, Strength, and Length")
        kind = "high" if match.group(1).lower() == "swinghigh" else "low"
        price_expr = args[1].strip()
        source_key = price_expr.lower()
        replacement = (
            f"swing_point('{kind}', '{source_key}', ({price_expr}), "
            f"{args[0].strip()}, {args[2].strip()}, {args[3].strip()})"
        )
        expr = expr[:match.start()] + replacement + expr[close_paren + 1:]
        search_from = match.start() + len(replacement)


def _rewrite_ofb_reference_calls(expr: str) -> str:
    """Pass PowerLanguage NumericRef output arguments by variable name."""
    patterns = (
        (re.compile(r"\bofb_cotpercentile\s*\(", re.I), 4, (2, 3), "OFB_CotPercentile"),
        (re.compile(r"\bvolumeprofilelevels\s*\(", re.I), 8, (5, 6, 7), "VolumeProfileLevels"),
    )
    search_from = 0
    while True:
        found = [(pattern.search(expr, search_from), expected, refs, name)
                 for pattern, expected, refs, name in patterns]
        found = [(match, expected, refs, name) for match, expected, refs, name in found if match]
        if not found:
            return expr
        match, expected, ref_indices, name = min(found, key=lambda row: row[0].start())
        open_paren = expr.find("(", match.start())
        depth, close_paren = 0, None
        for index in range(open_paren, len(expr)):
            if expr[index] == "(":
                depth += 1
            elif expr[index] == ")":
                depth -= 1
                if depth == 0:
                    close_paren = index
                    break
        if close_paren is None:
            return expr
        args = _split_declaration_parts(expr[open_paren + 1:close_paren])
        if len(args) != expected:
            if name == "OFB_CotPercentile":
                raise PowerLanguageError("OFB_CotPercentile expects SeriesName, IndexWeeks, oValid, and oNet")
            raise PowerLanguageError("VolumeProfileLevels expects five inputs and oPOC, oVAH, oVAL outputs")
        for index in ref_indices:
            if not re.fullmatch(r"[A-Za-z_]\w*", args[index].strip()):
                raise PowerLanguageError(f"{name} output arguments must be simple variable names")
            args[index] = repr(args[index].strip().lower())
        replacement = match.group(0).split("(", 1)[0].lower() + "(" + ", ".join(args) + ")"
        expr = expr[:match.start()] + replacement + expr[close_paren + 1:]
        search_from = match.start() + len(replacement)


@dataclass
class _Position:
    side: int
    qty: float
    entry: float
    entry_time: object
    entry_index: int
    max_favorable_price: float
    entry_commission: float


class _PLValue(float):
    def __new__(cls, value, field):
        obj = float.__new__(cls, value)
        obj.field = field
        return obj


def run_powerlanguage(data, source, asset_cfg, initial_cash, commission, slippage_ticks, progress_interval=3.0,
                      data2_timeframe="5min", session_timezone="UTC", num_contracts=None,
                      commission_rate_per_side=0.0, data2=None, ofb_provider=None):
    inputs, declared, tree = parse_source(source)
    program = compile_program(tree, set(inputs) | set(declared))
    program_names = set(program.co_names)
    requires_external_data2 = bool(re.search(
        r"(?im)^\s*//\s*ENGINE_DATA2_EXTERNAL_REQUIRED\s*=\s*true\s*$", source
    ))
    data2_names = {"barstatus2", "time_data2", "currentbar_data2", "open_data2", "high_data2", "low_data2",
                   "close_data2", "volume_data2", "open_data2_at", "high_data2_at",
                   "low_data2_at", "close_data2_at", "volume_data2_at", "xaverage_data2_at",
                   "average_data2_at", "stddev_data2_at", "avgtruerange_data2", "average_data2",
                   "xaverage_data2", "stddev_data2", "highest_data2", "lowest_data2", "summation_data2",
                   "date_data2"}
    uses_data2 = bool(data2_names.intersection(program_names))
    uses_rth_daily = bool({"highd", "lowd"}.intersection(program_names))
    if uses_data2 and requires_external_data2 and data2 is None:
        raise PowerLanguageError("this strategy requires a separate Data2 file; provide an implied-volatility OHLCV dataset")
    reserved_values = {f"value{i}": 0.0 for i in range(1, 100)}
    reserved_conditions = {f"condition{i}": False for i in range(1, 100)}
    variables = {**reserved_values, **reserved_conditions, **inputs, **declared}
    if num_contracts is not None:
        try:
            num_contracts = float(num_contracts)
        except (TypeError, ValueError) as exc:
            raise PowerLanguageError("NumContracts must be a positive number") from exc
        if not math.isfinite(num_contracts) or num_contracts <= 0:
            raise PowerLanguageError("NumContracts must be a positive finite number")
        variables["numcontracts"] = num_contracts
    point, tick = float(asset_cfg["point_value"]), float(asset_cfg["tick_size"])
    commission_rate_per_side = float(commission_rate_per_side)
    if not math.isfinite(commission_rate_per_side) or commission_rate_per_side < 0:
        raise PowerLanguageError("commission rate per side must be a nonnegative finite number")
    slip = float(slippage_ticks) * tick
    n = len(data)
    technical_indicators = IndicatorLibrary(data, session_timezone)
    ts = pd.DatetimeIndex(data.timestamp)
    try:
        session_ts = ts.tz_convert(session_timezone)
    except Exception as exc:
        raise PowerLanguageError(f"invalid session timezone {session_timezone!r}: {exc}") from exc
    # Data2 defaults to five-minute aggregation, matching the supplied strategy.
    # Buckets are right-closed so a timestamp at 08:35 completes 08:31–08:35.
    d2_period = pd.Timedelta(data2_timeframe)
    primary_period = ts.to_series().diff().median()
    if pd.isna(primary_period) or primary_period <= pd.Timedelta(0):
        primary_period = pd.Timedelta(minutes=1)
    # Input bars are open-stamped inside the simulator. PowerLanguage's Date
    # and Time values refer to the bar's closing timestamp.
    session_bar_close_ts = session_ts + primary_period
    close_years = session_bar_close_ts.year.to_numpy(dtype=np.int32)
    close_months = session_bar_close_ts.month.to_numpy(dtype=np.int32)
    close_days = session_bar_close_ts.day.to_numpy(dtype=np.int32)
    session_close_dates = close_years * 10000 + close_months * 100 + close_days
    session_close_minutes = (session_bar_close_ts.hour.to_numpy(dtype=np.int32) * 100
                             + session_bar_close_ts.minute.to_numpy(dtype=np.int32))
    del close_years, close_months, close_days
    # Infer session boundaries from material timestamp gaps when no explicit
    # exchange session template is supplied. SessionEndTime reads these values.
    session_end_minutes = session_close_minutes.copy()
    if n > 1:
        gaps = np.diff(ts.asi8)
        session_gap_threshold = max(primary_period * 1.5, pd.Timedelta(minutes=30))
        session_starts = np.r_[0, np.flatnonzero(gaps > session_gap_threshold.value) + 1]
        session_ends = np.r_[session_starts[1:] - 1, n - 1]
        for start_index, end_index in zip(session_starts, session_ends):
            session_end_minutes[start_index:end_index + 1] = session_close_minutes[end_index]
    if uses_data2 and data2 is not None:
        d2_source = data2.copy()
        d2_timestamps = pd.DatetimeIndex(d2_source.timestamp)
        try:
            d2_session_timestamps = d2_timestamps.tz_convert(session_timezone)
        except Exception as exc:
            raise PowerLanguageError(f"invalid Data2 session timezone {session_timezone!r}: {exc}") from exc
        d2_source.index = d2_session_timestamps
        d2_source_period = d2_session_timestamps.to_series().diff().median()
        if pd.isna(d2_source_period) or d2_source_period <= pd.Timedelta(0):
            d2_source_period = d2_period
        if d2_source_period >= d2_period * 0.8:
            # Data2 is already at the requested interval (or coarser) and is
            # open-stamped like Data1; its bar is available at the interval end.
            d2 = d2_source[["open", "high", "low", "close", "volume"]].copy()
            d2.index = d2.index + d2_period
        else:
            d2 = d2_source.resample(data2_timeframe, closed="right", label="right").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(subset=["open", "high", "low", "close"])
        d2series = {f"{field}_data2": d2[field].to_numpy(dtype=float)
                    for field in ("open", "high", "low", "close", "volume")}
        labels = pd.DatetimeIndex(d2.index)
        close_indices = np.searchsorted(labels.asi8, session_bar_close_ts.asi8, side="right") - 1
        close_indices[(close_indices < 0) | (close_indices >= len(labels))] = -1
        open_indices_d2 = np.searchsorted(labels.asi8, session_ts.asi8, side="right") - 1
        d2_bar_indices = np.where(close_indices > open_indices_d2, close_indices, -1).astype(np.int32)
        d2_current_indices = close_indices.astype(np.int32)
        d2_start_labels = (d2.index - pd.DateOffset(days=1)
                           if str(data2_timeframe).strip().lower() in {"1d", "d", "day", "daily"}
                           else d2.index - d2_period)
        d2_start_dates = (d2_start_labels.year.to_numpy(dtype=np.int32) * 10000
                          + d2_start_labels.month.to_numpy(dtype=np.int32) * 100
                          + d2_start_labels.day.to_numpy(dtype=np.int32))
        next_opens = labels + primary_period
        open_indices = ts.get_indexer(next_opens)
        missing = np.flatnonzero(open_indices < 0)
        if len(missing):
            open_indices[missing] = ts.searchsorted(next_opens[missing])
            open_indices[open_indices >= n] = -1
        d2_close_to_next_open = {
            pd.Timestamp(label): int(index) for label, index in zip(labels, open_indices) if index >= 0
        }
    elif uses_data2:
        d2_source = data.copy()
        d2_source.index = session_ts
        d2 = d2_source.resample(data2_timeframe, closed="right", label="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "high", "low", "close"])
        d2series = {f"{field}_data2": d2[field].to_numpy(dtype=float) for field in ("open", "high", "low", "close", "volume")}
        labels_in_primary_tz = d2.index.tz_convert(ts.tz) if d2.index.tz is not None and ts.tz is not None else d2.index
        labels = pd.DatetimeIndex(labels_in_primary_tz)
        if len(labels):
            insertions = np.searchsorted(labels.asi8, ts.asi8, side="left")
            valid_insertions = insertions < len(labels)
            matched = np.zeros(n, dtype=bool)
            matched[valid_insertions] = labels.asi8[insertions[valid_insertions]] == ts.asi8[valid_insertions]
            d2_bar_indices = np.where(matched, insertions, -1).astype(np.int32)
            del insertions, valid_insertions, matched
        else:
            d2_bar_indices = np.full(n, -1, dtype=np.int32)
        # For a primary bar timestamped at its open, choose the next Data2 label
        # as the in-progress higher-timeframe bar. Historical offset [1] then
        # correctly resolves to the latest fully closed Data2 bar.
        d2_current_indices = np.searchsorted(d2.index.asi8, ts.asi8, side="right").astype(np.int32)
        d2_start_labels = (d2.index - pd.DateOffset(days=1)
                           if str(data2_timeframe).strip().lower() in {"1d", "d", "day", "daily"}
                           else d2.index - d2_period)
        d2_start_dates = (d2_start_labels.year.to_numpy(dtype=np.int32) * 10000
                          + d2_start_labels.month.to_numpy(dtype=np.int32) * 100
                          + d2_start_labels.day.to_numpy(dtype=np.int32))
        # Vectorized alignment avoids constructing millions of pandas row objects.
        next_opens = labels + primary_period
        open_indices = ts.get_indexer(next_opens)
        missing = np.flatnonzero(open_indices < 0)
        if len(missing):
            open_indices[missing] = ts.searchsorted(next_opens[missing])
            open_indices[open_indices >= n] = -1
        d2_close_to_next_open = {
            pd.Timestamp(label): int(index) for label, index in zip(labels, open_indices) if index >= 0
        }
    else:
        # Data2 preparation includes a full resample and a per-bar alignment
        # index. Skip both when the compiled strategy has no Data2 references.
        d2_source = None
        d2 = pd.DataFrame(columns=("open", "high", "low", "close", "volume"))
        d2series = {f"{field}_data2": np.empty(0, dtype=float)
                    for field in ("open", "high", "low", "close", "volume")}
        d2_bar_indices = np.full(n, -1, dtype=np.int32)
        d2_current_indices = np.full(n, -1, dtype=np.int32)
        d2_start_dates = np.empty(0, dtype=np.int32)
        d2_close_to_next_open = {}
    series = {k: data[k].to_numpy(dtype=float, copy=False) for k in ("open", "high", "low", "close", "volume")}
    for tick_field in ("up_ticks", "down_ticks"):
        if tick_field in data:
            series[tick_field] = pd.to_numeric(data[tick_field], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    zero_ticks = np.zeros(n, dtype=np.float64)
    up_ticks = series.get("up_ticks", zero_ticks)
    down_ticks = series.get("down_ticks", zero_ticks)
    for quote_field in ("bid", "ask", "bid_size", "ask_size", "trade_count"):
        if quote_field in data:
            series[quote_field] = pd.to_numeric(data[quote_field], errors="coerce").to_numpy(dtype=np.float64)
    d2_partial = {}
    if uses_data2 and data2 is None and str(data2_timeframe).strip().lower() in {"1d", "d", "day", "daily"}:
        valid = (d2_current_indices >= 0) & (d2_current_indices < len(d2))
        buckets = d2_current_indices[valid]
        for field in ("open", "high", "low", "close", "volume"):
            values = np.full(n, np.nan, dtype=np.float64)
            frame = pd.DataFrame({"bucket": buckets, field: series[field][valid]})
            grouped = frame.groupby("bucket", sort=False)[field]
            if field == "open":
                partial = grouped.transform("first")
            elif field == "high":
                partial = grouped.cummax()
            elif field == "low":
                partial = grouped.cummin()
            elif field == "volume":
                partial = grouped.cumsum()
            else:
                partial = frame[field]
            values[valid] = partial.to_numpy(dtype=np.float64)
            d2_partial[f"{field}_data2"] = values
    for field in ("bid", "ask", "bid_size", "ask_size", "trade_count"):
        if uses_data2 and field in series and d2_source is not None and field in d2_source.columns:
            d2[field] = d2_source[field].resample(data2_timeframe, closed="right", label="right").last().reindex(d2.index).to_numpy(dtype=float)
            d2series[field + "_data2"] = d2[field].to_numpy(dtype=float)
    if uses_data2 and len(d2):
        # Build Data1-length, point-in-time Data2 series for multi-series
        # studies (Correlation, SpreadRatio, SpreadDiff). For external Data2,
        # each primary bar sees only the latest Data2 bar already completed;
        # locally aggregated daily Data2 uses its explicitly built partials.
        known_d2=np.searchsorted(d2.index.asi8,session_bar_close_ts.asi8,side="right")-1
        valid_d2=(known_d2>=0)&(known_d2<len(d2))
        for field in ("open","high","low","close","volume"):
            aligned=np.full(n,np.nan,dtype=np.float64)
            aligned[valid_d2]=d2[field].to_numpy(dtype=np.float64)[known_d2[valid_d2]]
            partial_values=d2_partial.get(f"{field}_data2")
            if partial_values is not None:
                aligned=np.where(np.isfinite(partial_values),partial_values,aligned)
            technical_indicators.fields[f"{field}_data2"]=aligned
    # HighD/LowD use the configured session timezone and RTH (09:30–16:00)
    # ranges. Current-day values are cumulative through the current bar, while
    # positive offsets address completed prior RTH sessions.
    if uses_rth_daily:
        session_dates = np.fromiter((int(x.strftime("%Y%m%d")) for x in session_ts), dtype=np.int64, count=n)
        session_minutes = (session_ts.hour.to_numpy(dtype=np.int32) * 100
                           + session_ts.minute.to_numpy(dtype=np.int32))
        rth_mask = (session_minutes >= 930) & (session_minutes < 1600)
        rth_indices = np.flatnonzero(rth_mask)
        rth_high_current = np.full(n, np.nan, dtype=np.float64)
        rth_low_current = np.full(n, np.nan, dtype=np.float64)
        if len(rth_indices):
            rth_rows = pd.DataFrame({
                "date": session_dates[rth_indices],
                "high": series["high"][rth_indices],
                "low": series["low"][rth_indices],
            })
            rth_groups = rth_rows.groupby("date", sort=True)
            rth_session_dates = rth_groups.size().index.to_numpy(dtype=np.int64)
            rth_daily_high = rth_groups["high"].max().to_numpy(dtype=np.float64)
            rth_daily_low = rth_groups["low"].min().to_numpy(dtype=np.float64)
            rth_high_current[rth_indices] = rth_groups["high"].cummax().to_numpy(dtype=np.float64)
            rth_low_current[rth_indices] = rth_groups["low"].cummin().to_numpy(dtype=np.float64)
        else:
            rth_session_dates = np.empty(0, dtype=np.int64)
            rth_daily_high = rth_daily_low = np.empty(0, dtype=np.float64)
    else:
        session_dates = session_minutes = np.empty(0, dtype=np.int64)
        rth_mask = np.empty(0, dtype=bool)
        rth_high_current = rth_low_current = np.empty(0, dtype=np.float64)
        rth_session_dates = np.empty(0, dtype=np.int64)
        rth_daily_high = rth_daily_low = np.empty(0, dtype=np.float64)

    def rth_daily_value(field, days_ago, bar_index, current_date, current_minute):
        ago = int(days_ago)
        if ago < 0 or not len(rth_session_dates):
            return 0.0
        slot = int(np.searchsorted(rth_session_dates, current_date, side="right") - 1)
        # Before today's RTH opens, today's session is not available yet.
        if current_minute < 930 and slot >= 0 and rth_session_dates[slot] == current_date:
            slot -= 1
        target = slot - ago
        if target < 0:
            return 0.0
        if ago == 0 and rth_mask[bar_index] and rth_session_dates[slot] == current_date:
            values = rth_high_current if field == "high" else rth_low_current
            return float(values[bar_index])
        values = rth_daily_high if field == "high" else rth_daily_low
        return float(values[target])
    # True range is shared by every requested AvgTrueRange period.
    prev_close = np.r_[np.nan, series["close"][:-1]]
    tr = np.nanmax(np.vstack([series["high"] - series["low"], np.abs(series["high"] - prev_close), np.abs(series["low"] - prev_close)]), axis=0)
    atr_cache = {}
    adx_cache = {}

    def atr_values(length):
        length = max(1, int(length))
        if length not in atr_cache:
            vals = np.full(n, np.nan, dtype=np.float64)
            if n >= length:
                vals[length - 1] = float(np.mean(tr[:length]))
                for j in range(length, n):
                    vals[j] = (vals[j - 1] * (length - 1) + tr[j]) / length
            atr_cache[length] = vals
        return atr_cache[length]

    def adx_values(length):
        length = max(1, int(length))
        if length not in adx_cache:
            high, low = series["high"], series["low"]
            up = np.r_[0.0, np.diff(high)]
            down = np.r_[0.0, -np.diff(low)]
            plus_dm = np.where((up > down) & (up > 0), up, 0.0)
            minus_dm = np.where((down > up) & (down > 0), down, 0.0)

            def wilder(values, period):
                result = np.full(len(values), np.nan, dtype=np.float64)
                if len(values) >= period:
                    result[period - 1] = float(np.mean(values[:period]))
                    for k in range(period, len(values)):
                        result[k] = (result[k - 1] * (period - 1) + values[k]) / period
                return result

            smooth_tr = wilder(tr, length)
            smooth_plus = wilder(plus_dm, length)
            smooth_minus = wilder(minus_dm, length)
            plus_di = np.full(len(smooth_tr), np.nan, dtype=np.float64)
            minus_di = np.full(len(smooth_tr), np.nan, dtype=np.float64)
            valid_tr = np.isfinite(smooth_tr) & (smooth_tr != 0)
            plus_di[valid_tr] = 100.0 * smooth_plus[valid_tr] / smooth_tr[valid_tr]
            minus_di[valid_tr] = 100.0 * smooth_minus[valid_tr] / smooth_tr[valid_tr]
            denom = plus_di + minus_di
            dx = np.full(len(denom), np.nan, dtype=np.float64)
            valid_denom = np.isfinite(denom) & (denom != 0)
            dx[valid_denom] = 100.0 * np.abs(plus_di[valid_denom] - minus_di[valid_denom]) / denom[valid_denom]
            result = np.full(len(dx), np.nan, dtype=np.float64)
            first = 2 * length - 2
            if len(dx) > first:
                result[first] = float(np.mean(dx[length - 1:first + 1]))
                for k in range(first + 1, len(dx)):
                    result[k] = (result[k - 1] * (length - 1) + dx[k]) / length
            adx_cache[length] = result
        return adx_cache[length]

    position = None
    cash = float(initial_cash)
    queued = None
    trades, eq, pos_curve = [], np.empty(n), np.zeros(n, dtype=np.float64)
    marketposition_history = None
    active_orders = []
    dollar_stop = dollar_target = None
    stop_per_contract = False
    risk_orders = {}
    exit_on_close = False
    plot_values = {}
    entryprice = 0.0
    barssinceentry = 0
    last_trade_pnl = 0.0
    next_progress = time.monotonic() + max(float(progress_interval), .25)
    next_progress_percent = 5.0
    indicator_cache = {}
    swing_cache = {}
    cross_state = {}
    variable_history = {}
    variable_ema_cache = {}
    derived_series = {}
    derived_ema_series = {}
    date_history, time_history = [], []

    def crosses(kind, current_x, current_y, key):
        previous = cross_state.get(key)
        cross_state[key] = (float(current_x), float(current_y))
        if previous is None:
            return False
        px, py = previous
        if kind == "over":
            return px <= py and float(current_x) > float(current_y)
        return px >= py and float(current_x) < float(current_y)

    def field_for(value, default="close"):
        return value.field if isinstance(value, _PLValue) else default

    def rolling(kind, field, length, index, ago=0):
        length = int(length)
        if length < 1:
            raise PowerLanguageError("indicator length must be positive")
        key = (kind, field, length)
        at_index = (int(d2_current_indices[index]) - int(ago) if field.endswith("_data2") else index - int(ago))
        if key not in indicator_cache:
            source_values = d2series[field] if field.endswith("_data2") else series[field]
            values = source_values
            source = pd.Series(values)
            if kind == "average": vals = source.rolling(length, min_periods=length).mean().to_numpy()
            elif kind == "xaverage": vals = source.ewm(span=length, adjust=False, min_periods=length).mean().to_numpy()
            elif kind == "highest": vals = source.rolling(length, min_periods=length).max().to_numpy()
            elif kind == "lowest": vals = source.rolling(length, min_periods=length).min().to_numpy()
            elif kind == "sum": vals = source.rolling(length, min_periods=length).sum().to_numpy()
            elif kind in {"stddev", "stddev_pop"}:
                ddof = 1 if kind == "stddev" else 0
                vals = source.rolling(length, min_periods=length).std(ddof=ddof).to_numpy(copy=True)
                means = source.rolling(length, min_periods=length).mean().to_numpy()
                floor = 64.0 * np.finfo(np.float64).eps * np.maximum(np.abs(means), 1.0)
                vals[np.isfinite(vals) & (vals <= floor)] = 0.0
            else: raise PowerLanguageError(f"unsupported indicator {kind}")
            indicator_cache[key] = vals
        return float(indicator_cache[key][at_index]) if 0 <= at_index < len(indicator_cache[key]) else np.nan

    def rolling_dynamic(kind, value, length, index):
        """Evaluate a variable-length OHLCV extrema over the current window."""
        try:
            length = int(length)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PowerLanguageError("dynamic indicator length must be an integer") from exc
        if length < 1:
            raise PowerLanguageError("indicator length must be positive")
        field = field_for(value)
        source_values = d2series.get(field) if field.endswith("_data2") else series.get(field)
        if source_values is None:
            raise PowerLanguageError(f"dynamic Highest/Lowest source is unavailable: {field}")
        at_index = int(d2_current_indices[index]) if field.endswith("_data2") else int(index)
        if at_index < length - 1 or at_index >= len(source_values):
            return np.nan
        window = source_values[at_index - length + 1:at_index + 1]
        kind = str(kind).lower()
        if kind in {"highest", "highestfc"}:
            return float(np.max(window))
        if kind in {"lowest", "lowestfc"}:
            return float(np.min(window))
        raise PowerLanguageError(f"dynamic lookback is supported for Highest/Lowest, not {kind}")

    def rolling_variable(name, current_value, kind, length, ago=0):
        length, ago = int(length), int(ago)
        if length < 1:
            raise PowerLanguageError("indicator length must be positive")
        if isinstance(current_value, _PLValue):
            field = field_for(current_value)
            return rolling(kind.lower(), field, length, i, ago)
        name, kind = name.lower(), kind.lower()
        history = variable_history.get(name, [])
        end = int(i) + 1 - ago
        if end < length:
            return np.nan
        start = end - length
        if kind == "xaverage":
            key = (name, length)
            state = variable_ema_cache.get(key)
            if state is None:
                state = {"values": np.full(n, np.nan, dtype=np.float64), "computed": -1}
                variable_ema_cache[key] = state
            values = state["values"]
            alpha = 2.0 / (length + 1.0)
            target = end - 1
            for bar in range(state["computed"] + 1, target + 1):
                value = float(current_value) if bar == i else float(history[bar])
                previous = values[bar - 1] if bar > 0 else np.nan
                values[bar] = value if bar == 0 or not np.isfinite(previous) else alpha * value + (1.0 - alpha) * previous
            state["computed"] = max(state["computed"], target)
            return float(values[target]) if target + 1 >= length else np.nan
        if end - 1 == i:
            prior = history[start:i]
            if len(prior) != length - 1:
                return np.nan
            sample = np.asarray(prior + [float(current_value)], dtype=np.float64)
        else:
            if end > len(history):
                return np.nan
            sample = np.asarray(history[start:end], dtype=np.float64)
        if kind == "average": return float(np.mean(sample))
        if kind == "highest": return float(np.max(sample))
        if kind == "lowest": return float(np.min(sample))
        raise PowerLanguageError(f"unsupported rolling indicator {kind}")

    def rolling_expression(kind, expression_key, current_value, length, data_type=1):
        length = int(length)
        if length < 1:
            raise PowerLanguageError("indicator length must be positive")
        values = derived_series.setdefault(expression_key, np.full(n, np.nan, dtype=np.float64))
        values[i] = float(current_value)
        if i + 1 < length:
            if kind == "xaverage":
                ema = derived_ema_series.setdefault((expression_key, length), np.full(n, np.nan, dtype=np.float64))
                ema[i] = values[i] if i == 0 or not np.isfinite(ema[i - 1]) else (
                    2.0 / (length + 1.0) * values[i] + (1.0 - 2.0 / (length + 1.0)) * ema[i - 1]
                )
            return np.nan
        sample = values[i - length + 1:i + 1]
        if not np.isfinite(sample).all():
            return np.nan
        if kind == "sum":
            return float(np.sum(sample))
        if kind == "average":
            return float(np.mean(sample))
        if kind == "highest":
            return float(np.max(sample))
        if kind == "lowest":
            return float(np.min(sample))
        if kind == "stddev":
            return float(np.std(sample, ddof=0 if int(data_type) == 1 else 1))
        if kind == "xaverage":
            ema = derived_ema_series.setdefault((expression_key, length), np.full(n, np.nan, dtype=np.float64))
            alpha = 2.0 / (length + 1.0)
            if i == 0:
                ema[i] = values[i]
            elif not np.isfinite(ema[i - 1]):
                ema[i] = values[i]
            else:
                ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]
            return float(ema[i])
        raise PowerLanguageError(f"unsupported derived-series indicator {kind}")

    def stddev_variable(name, current_value, length, data_type=1):
        length = int(length)
        if length < 1:
            raise PowerLanguageError("standard deviation length must be positive")
        history = variable_history.get(name.lower(), [])
        prior_count = length - 1
        if len(history) < prior_count:
            return np.nan
        sample = np.asarray((history[-prior_count:] if prior_count else []) + [float(current_value)], dtype=np.float64)
        return float(np.std(sample, ddof=0 if int(data_type) == 1 else 1))

    def rolling_data2(kind, field, length, bar_index):
        length = int(length)
        if length < 1:
            raise PowerLanguageError("indicator length must be positive")
        source_values = d2series.get(field)
        if source_values is None:
            return np.nan
        end = int(d2_current_indices[bar_index])
        if end < 0 or end >= len(source_values):
            return np.nan
        start = max(0, end - length + 1)
        window = np.asarray(source_values[start:end + 1], dtype=np.float64).copy()
        current_values = d2_partial.get(field)
        if current_values is not None and len(window):
            window[-1] = current_values[bar_index]
        if len(window) < length or not np.isfinite(window).all():
            return np.nan
        if kind == "average": return float(window.mean())
        if kind == "xaverage":
            alpha = 2.0 / (length + 1.0)
            result = float(window[0])
            for item in window[1:]: result = alpha * float(item) + (1.0 - alpha) * result
            return result
        if kind == "stddev": return float(window.std(ddof=1))
        if kind == "highest": return float(window.max())
        if kind == "lowest": return float(window.min())
        if kind == "sum": return float(window.sum())
        raise PowerLanguageError(f"unsupported Data2 indicator {kind}")

    def data2_atr_values(length):
        length = max(1, int(length))
        key = ("avgtruerange_data2", length)
        if key not in indicator_cache:
            h = d2["high"].to_numpy(dtype=np.float64)
            l = d2["low"].to_numpy(dtype=np.float64)
            c = d2["close"].to_numpy(dtype=np.float64)
            pc = np.r_[np.nan, c[:-1]]
            tr2 = np.nanmax(np.vstack([h-l, np.abs(h-pc), np.abs(l-pc)]), axis=0)
            vals = np.full(len(d2), np.nan, dtype=np.float64)
            if len(d2) >= length:
                vals[length-1] = float(np.mean(tr2[:length]))
                for j in range(length, len(d2)):
                    vals[j] = (vals[j-1] * (length-1) + tr2[j]) / length
            indicator_cache[key] = vals
        return indicator_cache[key]

    def rsi_series(field, length):
        length = int(length)
        if length < 1:
            raise PowerLanguageError("RSI length must be positive")
        key = ("rsi", field, length)
        if key not in indicator_cache:
            values = d2series[field] if field.endswith("_data2") else series[field]
            changes = np.diff(values, prepend=np.nan)
            gains = np.maximum(changes, 0.0)
            losses = np.maximum(-changes, 0.0)
            vals = np.full(len(values), np.nan, dtype=np.float64)
            if len(values) > length:
                avg_gain = float(np.mean(gains[1:length + 1]))
                avg_loss = float(np.mean(losses[1:length + 1]))
                for j in range(length, len(values)):
                    if j > length:
                        avg_gain = (avg_gain * (length - 1) + gains[j]) / length
                        avg_loss = (avg_loss * (length - 1) + losses[j]) / length
                    if avg_gain == 0 and avg_loss == 0:
                        vals[j] = 50.0
                    elif avg_loss == 0:
                        vals[j] = 100.0
                    elif avg_gain == 0:
                        vals[j] = 0.0
                    else:
                        vals[j] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            indicator_cache[key] = vals
        return indicator_cache[key]

    def pivot_series(kind, field, left_bars, right_bars):
        """Confirmed PowerLanguage pivot values, stamped on the confirmation bar."""
        left_bars, right_bars = int(left_bars), int(right_bars)
        if left_bars < 1 or right_bars < 1:
            raise PowerLanguageError("pivot left/right lengths must be positive")
        key = (kind, field, left_bars, right_bars)
        if key not in indicator_cache:
            values = d2series[field] if field.endswith("_data2") else series[field]
            width = left_bars + right_bars + 1
            result = np.full(len(values), np.nan, dtype=np.float64)
            if len(values) >= width:
                windows = np.lib.stride_tricks.sliding_window_view(values, width)
                center = windows[:, left_bars]
                left = windows[:, :left_bars]
                right = windows[:, left_bars + 1:]
                if kind == "pivothigh":
                    found = np.isfinite(center) & (center > np.max(left, axis=1)) & (center > np.max(right, axis=1))
                else:
                    found = np.isfinite(center) & (center < np.min(left, axis=1)) & (center < np.min(right, axis=1))
                # A pivot is only available after all right-side bars have closed.
                confirmation = np.arange(width - 1, len(values))
                result[confirmation[found]] = center[found]
            indicator_cache[key] = result
        return indicator_cache[key]

    def swing_point(kind, source_name, current_value, instance, strength, length, index):
        """Return the Nth most recent confirmed PowerLanguage swing price or -1."""
        try:
            instance, strength, length = int(instance), int(strength), int(length)
        except (TypeError, ValueError) as exc:
            raise PowerLanguageError("SwingHigh/SwingLow arguments must be numeric") from exc
        if instance < 1:
            raise PowerLanguageError("SwingHigh/SwingLow Instance must be at least 1")
        if strength < 1:
            raise PowerLanguageError("SwingHigh/SwingLow Strength must be at least 1")
        if length <= strength:
            raise PowerLanguageError("SwingHigh/SwingLow Length must exceed Strength")
        key = (kind, source_name, strength)
        state = swing_cache.get(key)
        if state is None:
            source_field = source_name.strip().lower()
            source_values = series.get(source_field)
            if source_values is None and source_field.endswith("_data2"):
                source_values = d2series.get(source_field)
            dynamic = source_values is None
            pivots = []
            if not dynamic and len(source_values) >= 2 * strength + 1:
                windows = np.lib.stride_tricks.sliding_window_view(source_values, 2 * strength + 1)
                center = windows[:, strength]
                left = windows[:, :strength]
                right = windows[:, strength + 1:]
                if kind == "low":
                    found = (np.isfinite(center) & np.isfinite(left).all(axis=1)
                             & np.isfinite(right).all(axis=1)
                             & (center <= np.min(left, axis=1)) & (center < np.min(right, axis=1)))
                else:
                    found = (np.isfinite(center) & np.isfinite(left).all(axis=1)
                             & np.isfinite(right).all(axis=1)
                             & (center >= np.max(left, axis=1)) & (center > np.max(right, axis=1)))
                # These candidates become visible only when their right-side
                # strength bars have closed; the query below enforces that.
                pivots = np.flatnonzero(found).astype(np.int64) + strength
            state = {
                "values": source_values if source_values is not None else np.full(n, np.nan, dtype=np.float64),
                "dynamic": dynamic,
                "pivots": pivots,
                "computed_until": -1,
                "query_index": -1,
                "query_start": 0,
                "query_end": 0,
            }
            swing_cache[key] = state
        if state["dynamic"]:
            state["values"][index] = float(current_value)
        values = state["values"]
        oldest_allowed = index - length + 1
        pivots = state["pivots"]
        if state["dynamic"]:
            while state["computed_until"] < index:
                completed_bar = state["computed_until"] + 1
                pivot_bar = completed_bar - strength
                if pivot_bar >= strength:
                    pivot_value = values[pivot_bar]
                    older = values[pivot_bar - strength:pivot_bar]
                    newer = values[pivot_bar + 1:pivot_bar + strength + 1]
                    if np.isfinite(pivot_value) and np.isfinite(older).all() and np.isfinite(newer).all():
                        if kind == "low":
                            confirmed = pivot_value <= np.min(older) and pivot_value < np.min(newer)
                        else:
                            confirmed = pivot_value >= np.max(older) and pivot_value > np.max(newer)
                        if confirmed:
                            pivots.append(pivot_bar)
                state["computed_until"] = completed_bar
        if state["query_index"] != index:
            previous_index = state["query_index"]
            if previous_index < 0 or index >= previous_index:
                query_start = state["query_start"]
                query_end = state["query_end"]
                while query_start < len(pivots) and pivots[query_start] < oldest_allowed:
                    query_start += 1
                confirmed_through = index - strength
                while query_end < len(pivots) and pivots[query_end] <= confirmed_through:
                    query_end += 1
            else:
                # Keep correct results for unusual callers that query bars out
                # of order; normal strategy execution advances one bar at a time.
                query_start = (bisect.bisect_left(pivots, oldest_allowed) if state["dynamic"] else
                               int(np.searchsorted(pivots, oldest_allowed, side="left")))
                query_end = (bisect.bisect_right(pivots, index - strength) if state["dynamic"] else
                             int(np.searchsorted(pivots, index - strength, side="right")))
            state["query_start"] = query_start
            state["query_end"] = query_end
            state["query_index"] = index
        selected = state["query_end"] - instance
        if selected < state["query_start"]:
            return -1.0
        return float(values[pivots[selected]])

    def commission_fee(price, qty):
        if commission_rate_per_side > 0:
            return abs(float(price) * point * float(qty)) * commission_rate_per_side
        return commission * float(qty)

    def exit_pos(i, price, reason="signal", apply_slippage=True, fill_time=None):
        nonlocal position, cash, last_trade_pnl
        if position is None:
            return
        fill = price - position.side * slip if apply_slippage else price
        gross = (fill - position.entry) * position.side * position.qty * point
        exit_commission = commission_fee(fill, position.qty)
        total_commission = position.entry_commission + exit_commission
        net = gross - total_commission
        cash += gross - exit_commission
        trades.append({"entry_time": position.entry_time, "exit_time": ts[i] if fill_time is None else fill_time, "side": position.side,
                       "quantity": position.qty, "entry_price": position.entry, "exit_price": fill,
                       "pnl": net, "commission": total_commission,
                       "entry_commission": position.entry_commission, "exit_commission": exit_commission,
                       "bars": i - position.entry_index, "exit_reason": reason})
        last_trade_pnl = net
        position = None

    def enter(i, side, qty, price, apply_slippage=True, fill_time=None):
        nonlocal position, cash, entryprice, barssinceentry
        if position is not None:
            exit_pos(i, price, "reversal")
        fill = price + side * slip if apply_slippage else price
        entry_commission = commission_fee(fill, qty)
        position = _Position(side, qty, fill, ts[i] if fill_time is None else fill_time, i, fill, entry_commission)
        entryprice, barssinceentry = fill, 0
        cash -= entry_commission

    def schedule_order(action, quantity, mode, price, this_close, this_open, line_no, order_label=""):
        nonlocal queued, active_orders
        try:
            qty = float(quantity)
            # The runner's NumContracts setting is an explicit entry-size
            # override. Exit instructions still close the simulated position.
            if num_contracts is not None and action.lower() in {"buy", "sellshort"}:
                qty = float(num_contracts)
            if not math.isfinite(qty) or qty <= 0:
                raise PowerLanguageError("order quantity must be positive")
            if this_close or this_open:
                if this_close:
                    fill_price = float(d2series["close_data2"][d2idx]) if barstatus2 else float(series["close"][i])
                else:
                    fill_price = float(d2series["open_data2"][d2idx]) if barstatus2 else float(series["open"][i])
                if action in ("buy", "sellshort"):
                    enter(i, 1 if action == "buy" else -1, qty, fill_price,
                          fill_time=session_bar_close_ts[i] if this_close else ts[i])
                elif position is not None:
                    exit_pos(i, fill_price, order_label or ("this_bar_close" if this_close else "this_bar_open"),
                             fill_time=session_bar_close_ts[i] if this_close else ts[i])
            elif mode == "market":
                if barstatus2:
                    next_open_index = d2_close_to_next_open.get(pd.Timestamp(ts[i]))
                    if next_open_index is None:
                        return  # No following Data2 bar: the next-bar order expires unfilled.
                else:
                    next_open_index = i + 1
                if next_open_index < n:
                    queued = (action, qty, next_open_index, order_label)
            else:
                if barstatus2:
                    exp = d2_close_to_next_open.get(pd.Timestamp(ts[i]))
                    if exp is None:
                        return  # No following Data2 bar: the next-bar order expires unfilled.
                    expiry = min(exp + int(d2_period / primary_period) - 1, n - 1)
                else:
                    expiry = min(i + 1, n - 1)
                active_orders.append((action, qty, mode, float(price), i, expiry, order_label))
        except Exception as exc:
            raise PowerLanguageError(f"line {line_no}: {exc}") from exc

    def set_stop(kind, value, line_no):
        nonlocal dollar_stop, dollar_target
        try:
            if kind == "loss": dollar_stop = float(value)
            else: dollar_target = float(value)
        except Exception as exc:
            raise PowerLanguageError(f"line {line_no}: invalid SetStop amount: {exc}") from exc

    def set_stop_pt(kind, value, line_no):
        try:
            set_stop(kind, float(value) * tick * point, line_no)
        except Exception as exc:
            if isinstance(exc, PowerLanguageError):
                raise
            raise PowerLanguageError(f"line {line_no}: invalid point-based SetStop amount: {exc}") from exc

    def set_stop_mode(per_contract):
        nonlocal stop_per_contract
        stop_per_contract = bool(per_contract)

    def set_risk_order(kind, value, line_no):
        try:
            values = value if isinstance(value, tuple) else (value,)
            point_based = kind.lower().endswith("_pt")
            base_kind = kind.lower().removesuffix("_pt")
            if base_kind == "setpercenttrailing" and len(values) != 2:
                raise ValueError("SetPercentTrailing expects floor and trail percent")
            if base_kind != "setpercenttrailing" and len(values) != 1:
                raise ValueError(f"{kind} expects one amount")
            amounts = [float(v) for v in values]
            if point_based:
                amounts[0] *= tick * point
            risk_orders[base_kind] = tuple(amounts)
        except Exception as exc:
            raise PowerLanguageError(f"line {line_no}: invalid {kind} amount: {exc}") from exc

    def set_exit_on_close(line_no):
        nonlocal exit_on_close
        exit_on_close = True

    def set_plot(plot_number, value, line_no):
        try:
            plot_number = int(plot_number)
            if plot_number < 1:
                raise ValueError("plot number must be positive")
            values = plot_values.setdefault(plot_number, np.full(n, np.nan, dtype=np.float64))
            value = float(value)
            values[i] = value if math.isfinite(value) else np.nan
        except Exception as exc:
            raise PowerLanguageError(f"line {line_no}: invalid Plot value: {exc}") from exc

    def clear_plot(plot_number, line_no):
        try:
            plot_number = int(plot_number)
            if plot_number < 1:
                raise ValueError("plot number must be positive")
            plot_values.setdefault(plot_number, np.full(n, np.nan, dtype=np.float64))[i] = np.nan
        except Exception as exc:
            raise PowerLanguageError(f"line {line_no}: invalid NoPlot value: {exc}") from exc

    def powerlanguage_exp(value):
        # PowerLanguage's exponential naturally saturates at the floating
        # point limit. Python's math.exp instead raises OverflowError, which
        # would abort an otherwise valid strategy (a common sigmoid edge case).
        value = float(value)
        try:
            return math.exp(value)
        except OverflowError:
            return math.inf if value > 0 else 0.0

    def while_limit(line_no):
        raise PowerLanguageError(f"line {line_no}: While loop exceeded 100,000 iterations on one bar")

    def series_at(name, ago):
        current = env.get(name, variables.get(name, 0))
        if isinstance(current, list):
            indices = ago if isinstance(ago, tuple) else (ago,)
            value = current
            try:
                for index in indices:
                    value = value[int(index)]
                return value
            except (IndexError, TypeError) as exc:
                raise PowerLanguageError(f"array {name} index {indices} is out of bounds") from exc
        ago = int(ago[0] if isinstance(ago, tuple) else ago)
        ago = max(0, ago)
        if ago == 0:
            return current
        hist = variable_history.get(name, ())
        return hist[-ago] if len(hist) >= ago else 0

    def array_set(array, indices, value, line_no=0):
        if not isinstance(array, list):
            raise PowerLanguageError(f"line {line_no}: indexed assignment target is not a declared Array")
        target = array
        try:
            for index in indices[:-1]:
                target = target[int(index)]
            target[int(indices[-1])] = value
        except (IndexError, TypeError) as exc:
            raise PowerLanguageError(f"line {line_no}: array index {indices} is out of bounds") from exc
        return value

    def array_set_max_index(array, max_index):
        if not isinstance(array, list):
            raise PowerLanguageError("Array_SetMaxIndex expects a one-dimensional dynamic array")
        max_index = int(max_index)
        if max_index < 0:
            raise PowerLanguageError("Array_SetMaxIndex must be nonnegative")
        if len(array) <= max_index:
            array.extend([0] * (max_index + 1 - len(array)))
        else:
            del array[max_index + 1:]
        return max_index

    def _array_range(array, start=0, end=None):
        if not isinstance(array, list) or any(isinstance(item, list) for item in array):
            raise PowerLanguageError("this array operation supports one-dimensional arrays only")
        start = int(start)
        end = len(array) - 1 if end is None else int(end)
        if start < 0 or end < start or end >= len(array):
            raise PowerLanguageError("array index range is out of bounds")
        return start, end

    def array_sum(array, *args):
        def flatten(values):
            for value in values:
                if isinstance(value, list):
                    yield from flatten(value)
                else:
                    yield float(value)
        if args:
            start, end = _array_range(array, args[0], args[1] if len(args) > 1 else None)
            return sum(flatten(array[start:end + 1]))
        return sum(flatten(array))

    def array_set_range(array, start, end, value):
        start, end = _array_range(array, start, end)
        array[start:end + 1] = [value] * (end - start + 1)
        return 0

    def array_fill(array, value):
        if not isinstance(array, list):
            raise PowerLanguageError("Fill_Array expects a declared Array")
        def fill(values):
            for index, item in enumerate(values):
                if isinstance(item, list):
                    fill(item)
                else:
                    values[index] = value
        fill(array)
        return 0

    def array_copy(source, source_index, destination, destination_index, count):
        if not isinstance(source, list) or not isinstance(destination, list):
            raise PowerLanguageError("Array_Copy expects declared arrays")
        source_index, destination_index, count = int(source_index), int(destination_index), int(count)
        if min(source_index, destination_index, count) < 0 or source_index + count > len(source):
            raise PowerLanguageError("Array_Copy source/destination range is out of bounds")
        # Snapshot first so overlapping copies match the source-before-write behavior.
        copied = list(source[source_index:source_index + count])
        required = destination_index + count
        if required > len(destination):
            destination.extend([0] * (required - len(destination)))
        destination[destination_index:required] = copied
        return 0

    def array_sort(array, start_index, end_index, ascending=True):
        if not isinstance(array, list) or any(isinstance(item, list) for item in array):
            raise PowerLanguageError("Array_Sort supports one-dimensional arrays only")
        start_index, end_index = _array_range(array, start_index, end_index)
        values = sorted(array[start_index:end_index + 1], reverse=not bool(ascending))
        array[start_index:end_index + 1] = values
        return 1

    # Keep history only for variables explicitly referenced with PowerLanguage
    # indexing (for example, MyVariable[1]). Retaining every scalar in the
    # per-bar environment made long minute-data runs consume several GB even
    # when the strategy never queried those histories.
    history_names = {
        value.lower()
        for value in program.co_consts
        if isinstance(value, str) and value.lower() in variables
    }
    needs_marketposition_history = "marketposition_at" in program_names
    needs_date_history = "date_at" in program_names
    needs_time_history = "time_at" in program_names
    if needs_marketposition_history:
        marketposition_history = np.zeros(n, dtype=np.int8)
    data_feature_prefixes = ("liq_", "flow_", "mbo_", "feed_", "option_", "ofb_")
    feature_columns = {str(c).strip().lower() for c in data.columns
                       if str(c).strip().lower().startswith(data_feature_prefixes)}
    ofb_helpers = {"ofb_getat", "ofb_poc", "ofb_vah", "ofb_val", "ofb_priorpoc",
                   "ofb_vwap", "ofb_vwap_sd", "ofb_biddepth", "ofb_askdepth",
                   "ofb_bidwallpx", "ofb_bidwallsz", "ofb_askwallpx", "ofb_askwallsz",
                   "ofb_bigtrade", "ofb_bigtradepx", "ofb_gexscalar", "ofb_gexat", "ofb_gexstrike",
                   "ofb_gexstrikes", "ofb_depthat", "ofb_depthlevels", "ofb_cot", "ofb_cotvalid",
                   "ofb_cotindex", "ofb_cotat", "ofb_cothas", "ofb_cotpercentile", "ofb_histcount",
                   "ofb_depthcount", "ofb_gexcount", "ofb_cotcount", "ofb_version", "ofb_status",
                   "ofb_agems", "ofb_loadhist", "ofb_loadgex", "ofb_loaddepth", "ofb_loadcot",
                   "ofb_gexcentroid", "ofb_gexmajpos", "ofb_gexmajneg", "ofb_gexcentroidday",
                   "ofb_gexposday", "ofb_gexnegday", "ofb_gexposoi", "ofb_gexnegoi", "ofb_gexbasis",
                   "ofb_gexdte", "ofb_gexflip", "ofb_gexflipday", "ofb_gexflipoi", "ofb_gexspot",
                   "ofb_gexflipsteep", "ofb_gexflipsteepday", "ofb_gexflipsteepoi"}
    required_features = {name for name in program.co_names
                         if name.startswith(data_feature_prefixes) and name not in ofb_helpers}
    missing_features = sorted(required_features - feature_columns)
    if missing_features:
        raise PowerLanguageError(
            "strategy data feature columns are unavailable in the selected feed: " + ", ".join(missing_features)
        )
    quote_names = {"bid", "ask", "bid_at", "ask_at", "insidebid", "insideask", "bidsize", "asksize",
                   "bidsize_at", "asksize_at", "tradecount", "tradecount_at"}
    required_quotes = quote_names.intersection(program.co_names)
    available_quotes = {name for name, col in (("bid", "bid"), ("ask", "ask"), ("bidsize", "bid_size"),
                                                 ("asksize", "ask_size"), ("tradecount", "trade_count")) if col in data.columns}
    if "bid" in available_quotes:
        available_quotes.update({"insidebid", "bid_at"})
    if "ask" in available_quotes:
        available_quotes.update({"insideask", "ask_at"})
    if "bidsize" in available_quotes:
        available_quotes.add("bidsize_at")
    if "asksize" in available_quotes:
        available_quotes.add("asksize_at")
    if "tradecount" in available_quotes:
        available_quotes.add("tradecount_at")
    # InsideBid/InsideAsk are commonly guarded by a positive-value check in
    # strategies so that OHLCV-only feeds can bypass spread filtering. The
    # per-bar environment supplies NaN when those optional quotes are absent.
    optional_quote_names = {"insidebid", "insideask"}
    missing_quotes = sorted(required_quotes - available_quotes - optional_quote_names)
    if missing_quotes:
        raise PowerLanguageError("strategy requires quote fields absent from this dataset: " + ", ".join(missing_quotes))
    referenced_tick_fields = {name for name in ("upticks", "downticks") if name in program.co_names}
    flow_source = inputs.get("flowsource")
    optional_tick_filter_disabled = inputs.get("usetickdelta") is False
    tick_mode_enabled = (bool(referenced_tick_fields) and not optional_tick_filter_disabled
                         and (flow_source is None or int(flow_source) == 2))
    if tick_mode_enabled:
        missing_ticks = sorted(name for name in referenced_tick_fields
                               if ("up_ticks" if name == "upticks" else "down_ticks") not in series)
        if missing_ticks:
            raise PowerLanguageError(
                "UpTicks/DownTicks require trade-tick history. Use a TBBO or MBO trade feed, "
                "or set FlowSource=1 for the OHLCV proxy. Missing: " + ", ".join(missing_ticks)
            )
    feature_series = {name: pd.to_numeric(data[name], errors="coerce").to_numpy(dtype=np.float64)
                      for name in feature_columns}
    safe_globals = {"__builtins__": {}}
    rng = np.random.default_rng(42)
    def raise_runtime_error(message, line_no):
        raise PowerLanguageError(f"line {line_no}: {message}")
    def get_last_trade_value(field):
        if not trades:
            return 0
        return trades[-1].get(field, 0)
    def closed_trade_datetime(field, part):
        value = get_last_trade_value(field)
        if not value:
            return 0
        value = pd.Timestamp(value)
        if value.tzinfo is not None:
            value = value.tz_convert(session_timezone)
        return int(value.strftime("%Y%m%d" if part == "date" else "%H%M"))
    env = {**variables, "__once_done": False, "range": range, "int": int,
           "__order__": schedule_order, "__set_stop__": set_stop,
           "__set_stop_pt__": set_stop_pt,
           "__set_stop_mode__": set_stop_mode, "__set_risk_order__": set_risk_order,
           "__set_exit_on_close__": set_exit_on_close, "__while_limit__": while_limit,
           "__plot__": set_plot, "__no_plot__": clear_plot,
           "__raise_runtime_error__": raise_runtime_error,
           "series_at": series_at,
           "__array_set__": array_set,
           "array_setmaxindex": array_set_max_index,
           "array_getmaxindex": lambda array: len(array) - 1,
           "array_setvalrange": array_set_range,
           "array_copy": array_copy,
           "fill_array": array_fill,
           "array_getfloatvalue": lambda array, index: array[int(index)],
           "array_getintegervalue": lambda array, index: int(array[int(index)]),
           "array_getbooleanvalue": lambda array, index: bool(array[int(index)]),
           "array_getstringvalue": lambda array, index: str(array[int(index)]),
           "array_setfloatvalue": lambda array, index, value: array_set(array, (index,), value),
           "array_setintegervalue": lambda array, index, value: array_set(array, (index,), int(value)),
           "array_setbooleanvalue": lambda array, index, value: array_set(array, (index,), bool(value)),
           "array_setstringvalue": lambda array, index, value: array_set(array, (index,), str(value)),
           "array_sum": array_sum,
           "array_sort": array_sort,
           "array_indexof": lambda array, value: (array.index(value) if value in array else -1),
           "array_contains": lambda array, value: value in array}

    profile_cache = None
    for i in range(n):
        d2idx = int(d2_bar_indices[i])
        barstatus2 = uses_data2 and d2idx >= 0
        if queued and i == queued[2]:
            action, qty, _, order_label = queued
            if action in ("buy", "sellshort"):
                enter(i, 1 if action == "buy" else -1, qty, float(series["open"][i]))
            elif position is not None:
                exit_pos(i, float(series["open"][i]), order_label or "market")
            queued = None

        # Process next-bar entry stop/limit orders while flat. Limit entries
        # fill at the open when already marketable, otherwise at their limit.
        if position is None and active_orders:
            remaining_orders = []
            entry_filled = False
            for action, order_qty, mode, price, placed, expires, order_label in active_orders:
                if action.lower() not in ("buy", "sellshort"):
                    remaining_orders.append((action, order_qty, mode, price, placed, expires, order_label))
                    continue
                if placed >= i or i > expires:
                    if i <= expires:
                        remaining_orders.append((action, order_qty, mode, price, placed, expires, order_label))
                    continue
                side = 1 if action.lower() == "buy" else -1
                fill_price = None
                if mode == "limit":
                    if side == 1 and series["open"][i] <= price:
                        fill_price = series["open"][i]
                    elif side == 1 and series["low"][i] <= price:
                        fill_price = price
                    elif side == -1 and series["open"][i] >= price:
                        fill_price = series["open"][i]
                    elif side == -1 and series["high"][i] >= price:
                        fill_price = price
                elif mode == "stop":
                    if side == 1 and series["open"][i] >= price:
                        fill_price = series["open"][i]
                    elif side == 1 and series["high"][i] >= price:
                        fill_price = price
                    elif side == -1 and series["open"][i] <= price:
                        fill_price = series["open"][i]
                    elif side == -1 and series["low"][i] <= price:
                        fill_price = price
                if fill_price is not None:
                    enter(i, side, order_qty, float(fill_price), apply_slippage=(mode == "stop"))
                    entry_filled = True
                else:
                    remaining_orders.append((action, order_qty, mode, price, placed, expires, order_label))
                if entry_filled:
                    break
            active_orders = remaining_orders

        # Apply active intrabar protective orders on the one-minute bars.
        if position is not None:
            side, qty = position.side, position.qty
            stop_price = target_price = None
            # PowerLanguage's default/SetStopPosition amount applies to the
            # whole position; SetStopContract/SetStopShare applies per unit.
            stop_denominator = point if stop_per_contract else point * qty
            if dollar_stop is not None:
                stop_price = position.entry - side * dollar_stop / stop_denominator
            if dollar_target is not None:
                target_price = position.entry + side * dollar_target / stop_denominator
            stop_candidates = []
            if stop_price is not None and ((side == 1 and series["low"][i] <= stop_price) or (side == -1 and series["high"][i] >= stop_price)):
                stop_candidates.append(("fixed_stop", stop_price))
            if target_price is not None and ((side == 1 and series["high"][i] >= target_price) or (side == -1 and series["low"][i] <= target_price)):
                # If both exits touch within the same 1-minute OHLC bar, stop is filled first.
                if not stop_candidates: stop_candidates.append(("fixed_target", target_price))
            trail_denominator = point if stop_per_contract else point * qty
            best_profit = max(0.0, (position.max_favorable_price - position.entry) * side * trail_denominator)
            break_even = risk_orders.get("setbreakeven")
            if break_even and best_profit >= break_even[0]:
                be_px = position.entry
                touched = (side == 1 and series["low"][i] <= be_px) or (side == -1 and series["high"][i] >= be_px)
                if touched and not stop_candidates: stop_candidates.append(("strategy_break_even", be_px))
            for trail_name in ("setdollartrailing", "settrailingstop", "setpercenttrailing"):
                trail = risk_orders.get(trail_name)
                if not trail:
                    continue
                if trail_name == "setpercenttrailing":
                    floor_amount, percent = trail
                    if percent < 0 or percent > 100:
                        raise PowerLanguageError("SetPercentTrailing percent must be between 0 and 100")
                    if best_profit < floor_amount:
                        continue
                    locked_profit = best_profit * (1.0 - percent / 100.0)
                else:
                    trail_amount = trail[0]
                    if best_profit <= trail_amount:
                        continue
                    locked_profit = best_profit - trail_amount
                trail_px = position.entry + side * locked_profit / trail_denominator
                touched = (side == 1 and series["low"][i] <= trail_px) or (side == -1 and series["high"][i] >= trail_px)
                if touched and not stop_candidates:
                    stop_candidates.append(("strategy_" + trail_name, trail_px))
            for action, order_qty, mode, price, placed, expires, order_label in active_orders:
                if placed >= i or i > expires or position is None: continue
                is_exit = (side == 1 and action.lower() == "sell") or (side == -1 and action.lower() == "buytocover")
                if not is_exit: continue
                hit = (mode == "stop" and ((side == 1 and series["low"][i] <= price) or (side == -1 and series["high"][i] >= price))) or (mode == "limit" and ((side == 1 and series["high"][i] >= price) or (side == -1 and series["low"][i] <= price)))
                if hit and not stop_candidates: stop_candidates.append(("strategy_" + mode + ("_" + order_label if order_label else ""), price))
            if stop_candidates:
                reason, price = stop_candidates[0]
                # Gap-through stop market orders fill at the bar open; limit targets fill at limit.
                apply_slippage = (reason == "fixed_stop" or reason == "strategy_stop" or
                                  reason.startswith("strategy_stop_") or "trailing" in reason or
                                  "break_even" in reason)
                if apply_slippage:
                    if side == 1 and series["open"][i] < price: price = series["open"][i]
                    if side == -1 and series["open"][i] > price: price = series["open"][i]
                else:
                    if side == 1 and series["open"][i] > price: price = series["open"][i]
                    if side == -1 and series["open"][i] < price: price = series["open"][i]
                exit_pos(i, float(price), reason, apply_slippage)
                active_orders = []

        if barstatus2:
            # Prior Data2 next-bar stop/limit orders expire at this completed bar.
            active_orders = []

        minute = int(session_close_minutes[i])
        date_num = int(session_close_dates[i])
        hhmm2 = minute if barstatus2 else 0
        d2_date_num = 0
        d2_current = int(d2_current_indices[i])
        if 0 <= d2_current < len(d2):
            d2_date_num = int(d2_start_dates[d2_current])
        env.update({
               "open": _PLValue(series["open"][i], "open"), "high": _PLValue(series["high"][i], "high"),
               "low": _PLValue(series["low"][i], "low"), "close": _PLValue(series["close"][i], "close"),
               "volume": _PLValue(series["volume"][i], "volume"),
               "upticks": float(up_ticks[i]),
               "downticks": float(down_ticks[i]),
               "open_data2": _PLValue(d2_partial["open_data2"][i], "open_data2") if "open_data2" in d2_partial else (_PLValue(d2series["open_data2"][d2idx], "open_data2") if barstatus2 else _PLValue(np.nan, "open_data2")),
               "high_data2": _PLValue(d2_partial["high_data2"][i], "high_data2") if "high_data2" in d2_partial else (_PLValue(d2series["high_data2"][d2idx], "high_data2") if barstatus2 else _PLValue(np.nan, "high_data2")),
               "low_data2": _PLValue(d2_partial["low_data2"][i], "low_data2") if "low_data2" in d2_partial else (_PLValue(d2series["low_data2"][d2idx], "low_data2") if barstatus2 else _PLValue(np.nan, "low_data2")),
               "close_data2": _PLValue(d2_partial["close_data2"][i], "close_data2") if "close_data2" in d2_partial else (_PLValue(d2series["close_data2"][d2idx], "close_data2") if barstatus2 else _PLValue(np.nan, "close_data2")),
               "volume_data2": _PLValue(d2_partial["volume_data2"][i], "volume_data2") if "volume_data2" in d2_partial else (_PLValue(d2series["volume_data2"][d2idx], "volume_data2") if barstatus2 else _PLValue(np.nan, "volume_data2")),
               "date": date_num, "date_data2": d2_date_num, "time": minute, "time_data2": hhmm2,
               "currentcontracts": float(position.qty) if position is not None else 0.0,
               "ticks": float(series["volume"][i]),
               "minmove": tick, "pricescale": 1.0,
               "barstatus1": 2, "barstatus2": 2 if barstatus2 else 0,
               "currentbar": i + 1, "currentbar_data2": int(d2_current_indices[i]) + 1,
               "currentdate": date_num, "currenttime": minute,
               "bartype": 1 if primary_period < pd.Timedelta(days=1) else 2,
               "bartype_ex": (2 if primary_period < pd.Timedelta(hours=1) else
                              (3 if primary_period < pd.Timedelta(days=1) else 4)),
               "barinterval": max(1, int(round(primary_period.total_seconds() / 60))) if primary_period < pd.Timedelta(days=1) else 1,
               "lastbaronchart": i == n - 1,
               "aioptimizing": 0, "getappinfo": lambda _app_info=0: 0,
               "random": lambda maximum=1: float(rng.random()) * max(float(maximum), 0.0),
               "numtostr": lambda value, decimals=0: format(float(value), f".{max(0, int(decimals))}f"),
               "timetominutes": lambda value: (int(value) // 100) * 60 + (int(value) % 100),
                "calctime": _calc_time,
               "sessionendtime": lambda *args: int(session_end_minutes[i]),
               "totaltrades": len(trades),
               "closed_entry_price": lambda: float(get_last_trade_value("entry_price")),
               "closed_exit_price": lambda: float(get_last_trade_value("exit_price")),
               "closed_entry_date": lambda: closed_trade_datetime("entry_time", "date"),
               "closed_entry_time": lambda: closed_trade_datetime("entry_time", "time"),
               "closed_exit_date": lambda: closed_trade_datetime("exit_time", "date"),
               "closed_exit_time": lambda: closed_trade_datetime("exit_time", "time"),
               "marketposition": 0 if position is None else position.side,
               "marketposition_at": lambda ago: int(marketposition_history[i-int(ago)]) if 0 <= i-int(ago) < n else 0,
               "marketposition_trade": lambda number: (int(position.side) if int(number) == 0 and position else
                                                          int(trades[-int(number)]["side"]) if int(number) > 0 and len(trades) >= int(number) else 0),
               "entryprice": entryprice, "barssinceentry": barssinceentry,
               "initialcapital": float(initial_cash), "netprofit": cash - float(initial_cash),
               "openpositionprofit": ((series["close"][i] - position.entry) * position.side * position.qty * point if position else 0.0),
               "currentequity": cash + ((series["close"][i] - position.entry) * position.side * position.qty * point if position else 0.0),
               "closedtradecount": len(trades), "lasttradepnl": last_trade_pnl,
               "commissionperunit": commission, "slippagepoints": slip, "pointvalue": point,
               "open_at": lambda ago: float(series["open"][max(0, i-int(ago))]) if i >= int(ago) else np.nan,
               "high_at": lambda ago: float(series["high"][max(0, i-int(ago))]) if i >= int(ago) else np.nan,
               "low_at": lambda ago: float(series["low"][max(0, i-int(ago))]) if i >= int(ago) else np.nan,
               "close_at": lambda ago: float(series["close"][max(0, i-int(ago))]) if i >= int(ago) else np.nan,
               "volume_at": lambda ago: float(series["volume"][max(0, i-int(ago))]) if i >= int(ago) else np.nan,
               "bid_at": lambda ago: float(series["bid"][max(0, i-int(ago))]) if "bid" in series and i >= int(ago) else np.nan,
               "ask_at": lambda ago: float(series["ask"][max(0, i-int(ago))]) if "ask" in series and i >= int(ago) else np.nan,
               "bidsize_at": lambda ago: float(series["bid_size"][max(0, i-int(ago))]) if "bid_size" in series and i >= int(ago) else np.nan,
               "asksize_at": lambda ago: float(series["ask_size"][max(0, i-int(ago))]) if "ask_size" in series and i >= int(ago) else np.nan,
               "tradecount_at": lambda ago: float(series["trade_count"][max(0, i-int(ago))]) if "trade_count" in series and i >= int(ago) else np.nan,
               "date_at": lambda ago: date_history[-int(ago)] if int(ago) > 0 and len(date_history) >= int(ago) else (date_num if int(ago) == 0 else 0),
               "time_at": lambda ago: time_history[-int(ago)] if int(ago) > 0 and len(time_history) >= int(ago) else (minute if int(ago) == 0 else 0),
               "open_data2_at": lambda ago: float(d2series["open_data2"][int(d2_current_indices[i])-int(ago)]) if 0 <= int(d2_current_indices[i])-int(ago) < len(d2) else np.nan,
               "high_data2_at": lambda ago: float(d2series["high_data2"][int(d2_current_indices[i])-int(ago)]) if 0 <= int(d2_current_indices[i])-int(ago) < len(d2) else np.nan,
               "low_data2_at": lambda ago: float(d2series["low_data2"][int(d2_current_indices[i])-int(ago)]) if 0 <= int(d2_current_indices[i])-int(ago) < len(d2) else np.nan,
               "close_data2_at": lambda ago: float(d2series["close_data2"][int(d2_current_indices[i])-int(ago)]) if 0 <= int(d2_current_indices[i])-int(ago) < len(d2) else np.nan,
               "volume_data2_at": lambda ago: float(d2series["volume_data2"][int(d2_current_indices[i])-int(ago)]) if 0 <= int(d2_current_indices[i])-int(ago) < len(d2) else np.nan,
               "avgtruerange": lambda length, ago=0: float(atr_values(length)[i - int(ago)]) if i >= int(ago) else np.nan,
               "adx": lambda length: float(adx_values(length)[i]),
               "adx_at": lambda length, ago=0: float(adx_values(length)[i - int(ago)]) if i >= int(ago) else np.nan,
               "xaverage_data2_at": lambda value, length, ago=0: rolling("xaverage", f"{field_for(value)}_data2", length, i, ago),
               "average_data2_at": lambda value, length, ago=0: rolling("average", f"{field_for(value)}_data2", length, i, ago),
               "stddev_data2_at": lambda value, length, ago=0: rolling("stddev", f"{field_for(value)}_data2", length, i, ago),
               "average_data2": lambda value, length: rolling_data2("average", field_for(value), length, i),
               "xaverage_data2": lambda value, length: rolling_data2("xaverage", field_for(value), length, i),
               "stddev_data2": lambda value, length: rolling_data2("stddev", field_for(value), length, i),
               "highest_data2": lambda value, length: rolling_data2("highest", field_for(value), length, i),
               "lowest_data2": lambda value, length: rolling_data2("lowest", field_for(value), length, i),
               "summation_data2": lambda value, length: rolling_data2("sum", field_for(value), length, i),
               "avgtruerange_data2": lambda length, ago=0: float(data2_atr_values(length)[int(d2_current_indices[i]) - int(ago)]) if 0 <= int(d2_current_indices[i]) - int(ago) < len(d2) else np.nan,
               "truerange": lambda: float(tr[i]),
               "crosses_over": lambda x, y, key: crosses("over", x, y, key),
               "crosses_under": lambda x, y, key: crosses("under", x, y, key),
               "average": lambda value, length: rolling("average", field_for(value), length, i),
               "averagefc": lambda value, length: rolling("average", field_for(value), length, i),
               "xaverage": lambda value, length: rolling("xaverage", field_for(value), length, i),
               "highestfc": lambda value, length: rolling("highest", field_for(value, "high"), length, i),
               "highest": lambda value, length: rolling("highest", field_for(value, "high"), length, i),
               "rolling_dynamic": lambda kind, value, length: rolling_dynamic(kind, value, length, i),
               "lowest": lambda value, length: rolling("lowest", field_for(value, "low"), length, i),
               "rolling_variable": lambda name, value, kind, length: rolling_variable(name, value, kind, length),
               "rolling_variable_at": lambda name, value, kind, length, ago=0: rolling_variable(name, value, kind, length, ago),
               "rolling_expression": rolling_expression,
               "highest_at": lambda value, length, ago=0: rolling("highest", field_for(value, "high"), length, i, ago),
               "lowest_at": lambda value, length, ago=0: rolling("lowest", field_for(value, "low"), length, i, ago),
               "average_at": lambda value, length, ago=0: rolling("average", field_for(value), length, i, ago),
               "xaverage_at": lambda value, length, ago=0: rolling("xaverage", field_for(value), length, i, ago),
               "summation": lambda value, length: rolling("sum", field_for(value), length, i),
               "stddev": lambda value, length, datatype=1: rolling("stddev_pop" if int(datatype) == 1 else "stddev", field_for(value), length, i),
               "stddev_variable": stddev_variable,
               "rsi": lambda value, length: float(rsi_series(field_for(value), length)[i]),
               "pivothigh": lambda value, left, right: float(pivot_series("pivothigh", field_for(value, "high"), left, right)[i]),
               "pivotlow": lambda value, left, right: float(pivot_series("pivotlow", field_for(value, "low"), left, right)[i]),
               "swing_point": lambda kind, source, value, instance, strength, length: swing_point(
                   kind, source, value, instance, strength, length, i),
               "isnan": lambda value: bool(pd.isna(value)),
               "bigpointvalue": point,
               "highd": lambda ago=0, bar_index=i, current_date=date_num, current_minute=minute: rth_daily_value("high", ago, bar_index, current_date, current_minute),
               "lowd": lambda ago=0, bar_index=i, current_date=date_num, current_minute=minute: rth_daily_value("low", ago, bar_index, current_date, current_minute),
               "dayofweek": lambda value: (_datetime.datetime.strptime(str(int(value)), "%Y%m%d").weekday() + 1) % 7,
               "datetojulian": lambda value: _datetime.datetime.strptime(str(int(value)), "%Y%m%d").date().toordinal(),
               "absvalue": abs, "min": min, "max": max, "square": lambda value: value * value,
               "minlist": min, "maxlist": max, "intportion": math.floor, "intportions": math.floor,
               "ceiling": math.ceil, "floor": math.floor, "log": math.log, "expvalue": powerlanguage_exp,
               "squareroot": math.sqrt, "sine": math.sin, "cosine": math.cos, "tangent": math.tan,
               "iff": lambda condition, when_true, when_false: when_true if condition else when_false,
               "iffstring": lambda condition, when_true, when_false: when_true if condition else when_false})
        indicator_names = {
            "sma", "simplemovingaverage", "movavgsimple", "movingaverage", "ema",
            "exponentialmovingaverage", "movavgexponential", "wma", "waverage",
            "weightedaverage", "weightedmovingaverage", "movavgweighted", "trima",
            "triangularmovingaverage", "movavgtriangular", "hma", "hullmovingaverage",
            "movavghull", "atr", "averagetruerange", "tr", "variance", "var",
            "wildersaverage", "wilderaverage", "exponentialaverage",
            "momentum", "mom", "roc", "rateofchange", "volumerateofchange",
            "relativestrengthindex", "cci", "commoditychannelindex", "commchannelindex",
            "plusdi", "diplus", "minusdi", "diminus", "adxr", "macd", "macdline",
            "macdsignal", "macdtrigger", "macdhist", "macdhistogram", "bollingerupper",
            "bollingerlower", "bollingerband", "bollingerbandupper", "bollingerbandlower",
            "bbupper", "bblower", "bollingerpercentb", "percentb", "bollingerbandwidth",
            "bandwidth", "fastk", "fastd", "slowk", "slowd", "stochastic", "stochasticfastk",
            "stochasticfastd", "stochasticslowk", "stochasticslowd", "williamsr", "percentr",
            "obv", "onbalancevolume", "accumdist", "accumulationdistribution", "chaikinad",
            "chaikinosc", "mfi", "moneyflowindex", "volumeoscillator", "volosc", "pvt",
            "pricevolumetrend", "trix", "ppo", "priceoscillator", "priceosc", "massindex",
            "ultosc", "ultimateoscillator", "ulcerindex", "ulcer", "aroonup", "aroondown",
            "donchianupper", "donchianlower", "donchianmid", "pricechannelhigh",
            "pricechannellow", "pricechannelmid", "keltnerupper", "keltnerlower", "keltnerchannel",
            "vwap", "volumeweightedaverageprice", "fisher", "fishertransform", "zscore",
            "standardscore", "linearregslope", "linearregvalue", "timeseriesforecast",
            "linearregline", "correlation", "correlationcoefficient", "kama", "parabolicsar",
            "parabolic", "stochrsi", "stochasticrsi", "aroonoscillator", "aroonosc",
            "choppiness", "choppinessindex", "chop", "dpo", "detrendedpriceoscillator",
            "easeofmovement", "emv", "forceindex", "force", "elderbullpower", "bullpower",
            "elderbearpower", "bearpower", "ichimokconversion", "ichimokubase",
            "ichimokuspana", "ichimokuspanb", "tenkansen", "kijunsen", "senkouspana",
            "senkouspanb", "relativevigorindex", "rvi",
            "typicalprice", "typical", "weightedclose", "weightedprice", "medianprice", "median",
            "awesomeoscillator", "ao", "acceleratoroscillator", "ac", "priceenvelopeupper",
            "priceenvelopelower", "movavgenvelopeupper", "movavgenvelopelower", "standarderror",
            "stderr", "standarderrorupper", "standarderrorlower", "doji", "insidebar", "outsidebar",
            "bullengulfing", "bearengulfing", "hammer", "shootingstar", "morningstar", "eveningstar",
            "gapup", "gapdown",
            "bullharami", "bearharami", "hangingman", "morningdoji", "eveningdoji", "piercingline",
            "darkcloudcover", "threewhitesoldiers", "threeblackcrows", "keyreversalup", "keyreversaldown",
            "percentchange", "pctchange", "percentincrease", "percentdecrease",
            "averageclosegreateropen", "avgclosegreateropen", "avgcloselessthanavgopen",
            "avgclosegreaterthanavgopen", "averageclosegreaterthanaverageopen",
            "averagecloselessthanaverageopen", "pricegreaterthanaverage", "pricelessthanaverage",
            "bidaskspread", "spread", "bidaskmidpoint", "quoteprice", "bidaskvolratio",
            "bidaskvolumeratio", "marketfacilitationindex", "marketfacilitation", "mfi_bar",
            "marketdepthbidprice", "marketdepthaskprice", "marketdepthbidsize", "marketdepthasksize",
            "marketdepthbid", "marketdepthask", "marketdepthimbalance", "marketdepthspread", "marketdepthmicroprice",
            "marketdepthbboimbalance", "marketdepthtopimbalance", "marketdepthweightedmid", "marketdepthweightedmidprice",
            "volumeaverage", "volumeavg", "volumeratio", "volumebreakout", "volumegradient",
            "volumeup", "volumedown", "volumetoday", "volumecumulative", "dayopen", "dayhigh", "daylow",
            "firstbarofday", "firstbarofweek", "firstbarofmonth", "firstbarofyear",
            "newhighforchart", "newlowforchart", "newhighforday", "newlowforday",
            "newhighforweek", "newlowforweek", "newhighformonth", "newlowformonth",
            "newhighforyear", "newlowforyear", "newndayhigh", "newndaylow", "breakoutofxbarhigh",
            "breakoutofxbarlow", "floortraderpivot", "floortraderpivots", "pivotpoint",
            "floortraderresistance1", "floortradersupport1", "yesterdayhigh", "yesterdaylow",
            "yesterdayclose", "accumdistbuypr", "accumdistprvol", "moneyflow", "volatility",
            "volatilitystddev", "consecutiveups", "consecutiveup", "consecutivedowns", "consecutivedown",
            "nextthirdfriday", "expiration", "swingindex", "accumswingindex", "accumulativeswingindex",
            "pivothigh", "pivotlow", "zigzagpercent", "zigzagpct", "zigzagpoints", "zigzagpoint",
            "momentumincreasing", "momentumdecreasing", "cciaverage", "ccisma", "islandreversalup",
            "islandreversaldown", "spreadratio", "spreaddiff", "spread_diff",
            "alligatorjaw", "alligatorteeth", "alligatorlips", "chaosalligatorjaw",
            "chaosalligatorteeth", "chaosalligatorlips", "gatorupper", "gatorlower",
            "chaosgatorupper", "chaosgatorlower", "chaosfractalhigh", "chaosfractallow",
            "newintradayhigh", "newintradaylow", "newintradayhilo",
            "marneyvolume", "marneyvolumeindicator", "marneyrange", "marneyrangeindicator",
        }
        for alias in INDICATOR_ALIASES:
            if " " not in alias:
                indicator_names.add(alias)
        for indicator_name in indicator_names:
            if indicator_name not in env:
                canonical_name = INDICATOR_ALIASES.get(indicator_name, indicator_name)
                env[indicator_name] = (
                    lambda *args, _name=canonical_name: technical_indicators.value(_name, i, *args)
                )
        for called_name in program_names:
            if called_name.startswith(("tl_", "text_", "arw_", "rectangle_", "ellipse_",
                                       "trendline_", "verticalline_", "horizontalline_")):
                env.setdefault(called_name, chart_noop)
        for input_name, input_default in inputs.items():
            if (isinstance(input_default, tuple) and len(input_default) == 2
                    and input_default[0] == "__series_default__"):
                field = input_default[1]
                env[input_name] = _PLValue(float(series[field][i]), field)
        for quote_field in ("bid", "ask", "bid_size", "ask_size", "trade_count"):
            if quote_field in series:
                env[quote_field] = _PLValue(series[quote_field][i], quote_field)
        env["insidebid"] = env.get("bid", np.nan)
        env["insideask"] = env.get("ask", np.nan)
        env["last"] = env["close"]
        env["bidsize"] = env.get("bid_size", np.nan)
        env["asksize"] = env.get("ask_size", np.nan)
        env["tradecount"] = env.get("trade_count", np.nan)
        env.update({name: float(values[i]) for name, values in feature_series.items()})
        # OrderFlowBridge functions backed by the bundled timestamp-keyed
        # feature files. Slots are loaded by run_backtest.py only when the
        # strategy references these helpers.
        current_ofb_key = int(session_bar_close_ts[i].strftime("%Y%m%d%H%M%S"))
        input_ofb_key = int(session_ts[i].strftime("%Y%m%d%H%M%S"))
        def ofb_key_candidates(key):
            base_key = current_ofb_key if key is None or int(float(key)) <= 0 else int(float(key))
            candidates = [base_key]
            if base_key == current_ofb_key:
                candidates.extend((input_ofb_key,))
            try:
                base_time = pd.to_datetime(f"{base_key:014d}", format="%Y%m%d%H%M%S")
                candidates.extend((int((base_time - primary_period).strftime("%Y%m%d%H%M%S")),
                                  int((base_time + primary_period).strftime("%Y%m%d%H%M%S"))))
            except (TypeError, ValueError, OverflowError):
                pass
            return tuple(dict.fromkeys(candidates))
        def ofb_slot(slot, prefix="ofb_hist", key=None):
            name = f"{prefix}_{int(slot)}"
            if key is None or int(float(key)) <= 0 or int(float(key)) == current_ofb_key:
                if name in feature_series:
                    return float(feature_series[name][i])
                key = current_ofb_key
            if ofb_provider is not None and prefix == "ofb_hist":
                for candidate in ofb_key_candidates(key):
                    value = ofb_provider.hist_at(slot, candidate)
                    if np.isfinite(value):
                        return value
            if ofb_provider is not None and prefix == "ofb_gex":
                for candidate in ofb_key_candidates(key):
                    value = ofb_provider.gex_scalar(slot, candidate)
                    if np.isfinite(value):
                        return value
            return np.nan
        poc_slots = {1: 24, 2: 18, 3: 20, 4: 22, 5: 26, 6: 66, 7: 70}
        vah_slots = {1: 28, 2: 34, 3: 64, 4: 30, 5: 32, 6: 68, 7: 72}
        val_slots = {1: 29, 2: 35, 3: 65, 4: 31, 5: 33, 6: 69, 7: 73}
        prior_poc_slots = {1: 25, 2: 19, 3: 21, 4: 23, 5: 27, 6: 67, 7: 71}
        vwap_slots = {1: 42, 2: 36, 3: 38, 4: 40, 5: 44}
        vwap_sd_slots = {1: 43, 2: 37, 3: 39, 4: 41, 5: 45}
        def ofb_cot(name, key=0):
            column = f"ofb_cot_{str(name).strip().lower()}"
            value = feature_series.get(column)
            if key is None or int(float(key)) <= 0 or int(float(key)) == current_ofb_key:
                if value is not None and not pd.isna(value[i]):
                    return float(value[i])
                key = current_ofb_key
            if ofb_provider is not None:
                return ofb_provider.cot_at(ofb_provider.cot_index(name), key)
            return 0.0
        def ofb_cot_valid(name, key=0):
            if ofb_provider is not None:
                return ofb_provider.cot_has(ofb_provider.cot_index(name), key or current_ofb_key)
            column = f"ofb_cot_{str(name).strip().lower()}"
            value = feature_series.get(column)
            return int(value is not None and not pd.isna(value[i]))
        def ofb_cot_percentile(name, weeks, valid_ref, net_ref):
            if ofb_provider is None:
                percentile, valid, net = 50.0, 0, 0.0
            else:
                percentile, valid, net = ofb_provider.cot_percentile(name, weeks, current_ofb_key)
            env[str(valid_ref).lower()] = valid
            env[str(net_ref).lower()] = net
            return percentile
        def ofb_gex_scalar(slot, key=0):
            return ofb_slot(slot, "ofb_gex", key if key else current_ofb_key)
        def ofb_load(name):
            if ofb_provider is None:
                return 0
            return int(getattr(ofb_provider, name)() > 0)
        def ofb_gex_at(rank, what, key=0):
            if ofb_provider is None:
                return np.nan
            return next((value for candidate in ofb_key_candidates(key)
                         if np.isfinite(value := ofb_provider.gex_at(rank, what, candidate))), np.nan)
        def ofb_depth_at(side, level, what, key=0):
            if ofb_provider is None:
                return np.nan
            return next((value for candidate in ofb_key_candidates(key)
                         if np.isfinite(value := ofb_provider.depth_at(side, level, what, candidate))), np.nan)
        def ofb_gex_func(slot):
            return lambda key=0: ofb_gex_scalar(slot, key)
        chart_noop = lambda *args, **kwargs: 0
        def num_to_str(value, decimals=0):
            return f"{float(value):.{max(0, min(12, int(decimals)))}f}"
        def volume_profile_levels(window_bars, use_tick_rows, ticks_per_row, num_rows,
                                  value_area_pct, poc_ref, vah_ref, val_ref):
            nonlocal profile_cache
            profile_cache = profile_cache or {}
            cache_key = (i, int(window_bars), bool(use_tick_rows), int(ticks_per_row),
                         int(num_rows), float(value_area_pct))
            if cache_key in profile_cache:
                ok, poc, vah, val = profile_cache[cache_key]
            else:
                count = int(float(window_bars))
                ok, poc, vah, val = 0, 0.0, 0.0, 0.0
                start = i - count + 1
                if count >= 2 and start >= 0 and i + 1 > count:
                    hi = float(np.max(series["high"][start:i + 1]))
                    lo = float(np.min(series["low"][start:i + 1]))
                    total_range = hi - lo
                    rows_fallback = max(2, min(500, int(float(num_rows))))
                    if total_range > 0:
                        if use_tick_rows:
                            tick_step = tick * int(float(ticks_per_row))
                            if tick_step <= 0:
                                tick_step = total_range / rows_fallback
                            if tick > 0 and total_range / tick_step > 498:
                                tick_step += math.ceil((total_range / 498 - tick_step) / tick) * tick
                            lo = math.floor(lo / tick_step) * tick_step
                            row_count = int((hi - lo) / tick_step) + 1
                        else:
                            row_count = rows_fallback
                            tick_step = total_range / row_count
                        row_count = max(2, min(500, int(row_count)))
                        if tick_step > 0 and math.isfinite(tick_step):
                            bins = np.zeros(row_count, dtype=np.float64)
                            for bar in range(start, i + 1):
                                low_row = max(0, int((float(series["low"][bar]) - lo) / tick_step))
                                high_row = min(row_count - 1, int((float(series["high"][bar]) - lo) / tick_step))
                                if high_row < low_row:
                                    high_row = low_row
                                span = high_row - low_row + 1
                                contribution = float(series["volume"][bar]) / span
                                bins[low_row:high_row + 1] += contribution
                            total_vol = float(np.sum(bins))
                            max_vol = float(np.max(bins))
                            if total_vol > 0 and max_vol > 0:
                                poc_row = int(np.argmax(bins))
                                upper = lower = poc_row
                                accumulated = float(bins[poc_row])
                                target = total_vol * float(value_area_pct)
                                while accumulated < target and (upper < row_count - 1 or lower > 0):
                                    up_vol = float(bins[upper + 1]) if upper < row_count - 1 else -1.0
                                    dn_vol = float(bins[lower - 1]) if lower > 0 else -1.0
                                    if up_vol >= dn_vol:
                                        upper += 1
                                        accumulated += max(0.0, up_vol)
                                    else:
                                        lower -= 1
                                        accumulated += max(0.0, dn_vol)
                                poc = lo + (poc_row + 0.5) * tick_step
                                vah = lo + (upper + 1) * tick_step
                                val = lo + lower * tick_step
                                ok = 1
                profile_cache.clear()
                profile_cache[cache_key] = (ok, poc, vah, val)
            env[str(poc_ref).lower()] = poc
            env[str(vah_ref).lower()] = vah
            env[str(val_ref).lower()] = val
            return ok
        env.update({
            "ofb_getat": lambda slot, key=0: ofb_slot(slot, "ofb_hist", key if key else current_ofb_key),
            "ofb_get": lambda slot: ofb_slot(slot, "ofb_hist"),
            "ofb_poc": lambda mode=1: ofb_slot(poc_slots.get(int(mode), 24)),
            "ofb_vah": lambda mode=1: ofb_slot(vah_slots.get(int(mode), 0)) if int(mode) in vah_slots else np.nan,
            "ofb_val": lambda mode=1: ofb_slot(val_slots.get(int(mode), 0)) if int(mode) in val_slots else np.nan,
            "ofb_priorpoc": lambda mode=1: ofb_slot(prior_poc_slots.get(int(mode), 25)),
            "ofb_vwap": lambda mode=1: ofb_slot(vwap_slots.get(int(mode), 0)) if int(mode) in vwap_slots else np.nan,
            "ofb_vwap_sd": lambda mode=1: ofb_slot(vwap_sd_slots.get(int(mode), 0)) if int(mode) in vwap_sd_slots else np.nan,
            "ofb_biddepth": lambda levels=5: ofb_slot(52 if int(levels) >= 50 else 50 if int(levels) >= 20 else 48 if int(levels) >= 10 else 3),
            "ofb_askdepth": lambda levels=5: ofb_slot(53 if int(levels) >= 50 else 51 if int(levels) >= 20 else 49 if int(levels) >= 10 else 4),
            "ofb_bidwallpx": lambda rank=1: ofb_slot(56 if int(rank) >= 2 else 54),
            "ofb_bidwallsz": lambda rank=1: ofb_slot(57 if int(rank) >= 2 else 55),
            "ofb_askwallpx": lambda rank=1: ofb_slot(60 if int(rank) >= 2 else 58),
            "ofb_askwallsz": lambda rank=1: ofb_slot(61 if int(rank) >= 2 else 59),
            "ofb_bigtrade": lambda: ofb_slot(46),
            "ofb_bigtradepx": lambda: ofb_slot(47),
            "ofb_gexscalar": ofb_gex_scalar,
            "ofb_gexat": ofb_gex_at,
            "ofb_gexstrike": ofb_gex_at,
            "ofb_gexstrikes": lambda: 128 if ofb_provider is not None and ofb_provider.gex_count() else 0,
            "ofb_depthat": ofb_depth_at,
            "ofb_depthlevels": lambda: 64 if ofb_provider is not None and ofb_provider.depth_count() else 0,
            "ofb_cot": ofb_cot,
            "ofb_cotvalid": ofb_cot_valid,
            "ofb_cotindex": lambda name: ofb_provider.cot_index(name) if ofb_provider is not None else -1,
            "ofb_cotat": lambda index, key=0: ofb_provider.cot_at(index, key or current_ofb_key) if ofb_provider is not None else 0.0,
            "ofb_cothas": lambda index, key=0: ofb_provider.cot_has(index, key or current_ofb_key) if ofb_provider is not None else 0,
            "ofb_cotpercentile": ofb_cot_percentile,
            "ofb_histcount": lambda: ofb_provider.hist_count() if ofb_provider is not None else 0,
            "ofb_depthcount": lambda: ofb_provider.depth_count() if ofb_provider is not None else 0,
            "ofb_gexcount": lambda: ofb_provider.gex_count() if ofb_provider is not None else 0,
            "ofb_cotcount": lambda: (len(ofb_provider.cot_data()[0]) if ofb_provider is not None and ofb_provider.cot_data() is not None else 0),
            "ofb_version": lambda: 10300 if ofb_provider is not None else 0,
            "ofb_status": lambda: int(ofb_provider is not None),
            "ofb_agems": lambda: 0,
            "ofb_loadhist": lambda *args: ofb_load("hist_count"),
            "ofb_loadgex": lambda *args: ofb_load("gex_count"),
            "ofb_loaddepth": lambda *args: ofb_load("depth_count"),
            "ofb_loadcot": lambda *args: int(ofb_provider is not None and ofb_provider.cot_data() is not None),
            "ofb_gexcentroid": ofb_gex_func(0), "ofb_gexmajpos": ofb_gex_func(1),
            "ofb_gexmajneg": ofb_gex_func(2), "ofb_gexcentroidday": ofb_gex_func(3),
            "ofb_gexposday": ofb_gex_func(4), "ofb_gexnegday": ofb_gex_func(5),
            "ofb_gexposoi": ofb_gex_func(6), "ofb_gexnegoi": ofb_gex_func(7),
            "ofb_gexbasis": ofb_gex_func(8), "ofb_gexdte": ofb_gex_func(9),
            "ofb_gexflip": ofb_gex_func(10), "ofb_gexflipday": ofb_gex_func(11),
            "ofb_gexflipoi": ofb_gex_func(12), "ofb_gexspot": ofb_gex_func(13),
            "ofb_gexflipsteep": ofb_gex_func(14), "ofb_gexflipsteepday": ofb_gex_func(15),
            "ofb_gexflipsteepoi": ofb_gex_func(16),
            "volumeprofilelevels": volume_profile_levels,
            "minmove": tick, "pricescale": 1,
            "rgb": lambda red, green, blue: (int(red) & 255) | ((int(green) & 255) << 8) | ((int(blue) & 255) << 16),
            "adddate": lambda value, days: int((_datetime.datetime.strptime(str(int(value)), "%Y%m%d")
                                                  + _datetime.timedelta(days=int(days))).strftime("%Y%m%d")),
            "numtostr": num_to_str,
            "lastbaronchart_s": bool(i == n - 1), "lastbaronchart": bool(i == n - 1),
            "tl_new": chart_noop, "tl_new_s": chart_noop, "tl_setcolor": chart_noop,
            "tl_setstyle": chart_noop, "tl_setsize": chart_noop, "tl_setend": chart_noop,
            "tl_setend_s": chart_noop, "tl_setextleft": chart_noop, "tl_setextright": chart_noop,
            "tl_delete": chart_noop, "text_new": chart_noop, "text_new_s": chart_noop,
            "text_setcolor": chart_noop, "text_setstyle": chart_noop, "text_setstring": chart_noop,
            "text_setlocation": chart_noop, "text_delete": chart_noop,
            "arw_new": chart_noop, "arw_setcolor": chart_noop, "arw_setsize": chart_noop,
            "rectangle_new_bn": chart_noop, "rectangle_setpattern": chart_noop,
            "rectangle_setstyle": chart_noop, "rectangle_setsize": chart_noop,
            "rectangle_setfillcolor": chart_noop, "rectangle_setcolor": chart_noop,
            "alert": chart_noop, "setplotcolor": chart_noop, "setplotwidth": chart_noop,
            "setplotbgcolor": chart_noop, "setplotstyle": chart_noop,
            "tool_solid": 0, "tool_dashed": 1, "tool_dotted": 2,
            "red": 255, "green": 32768, "blue": 16711680,
            "yellow": 65535, "white": 16777215, "black": 0,
            "julian_to_date": lambda value: int(_datetime.date.fromordinal(int(value)).strftime("%Y%m%d")),
        })
        try:
            exec(program, safe_globals, env)
        except PowerLanguageError:
            raise
        except Exception as exc:
            raise PowerLanguageError(f"runtime error at bar {i + 1} ({ts[i]}): {exc}") from exc
        # Update favorable excursion only after evaluating this bar's protective
        # orders. This avoids using an unknown high/low sequence to activate a
        # trailing stop and trigger it within the same OHLC bar.
        if position is not None:
            favorable = float(series["high"][i]) if position.side == 1 else float(series["low"][i])
            position.max_favorable_price = max(position.max_favorable_price, favorable) if position.side == 1 else min(position.max_favorable_price, favorable)
        if exit_on_close and position is not None:
            session_end = i + 1 >= n or (ts[i + 1] - ts[i]) > primary_period * 1.5
            if session_end:
                exit_pos(i, float(series["close"][i]), "set_exit_on_close", fill_time=session_bar_close_ts[i])
                active_orders = []
        for name in history_names:
            value = env.get(name)
            if callable(value):
                continue
            if isinstance(value, (bool, int, float, np.number, str)):
                variable_history.setdefault(name, []).append(value)
        if needs_date_history:
            date_history.append(date_num)
        if needs_time_history:
            time_history.append(minute)
        if position is not None: barssinceentry = i - position.entry_index
        mtm = cash + ((series["close"][i] - position.entry) * position.side * position.qty * point if position else 0.0)
        eq[i] = mtm
        pos_curve[i] = 0 if position is None else position.side * position.qty
        if marketposition_history is not None:
            marketposition_history[i] = 0 if position is None else position.side
        if progress_interval > 0:
            current_progress_percent = (i + 1) * 100.0 / n
            if current_progress_percent >= next_progress_percent or time.monotonic() >= next_progress:
                print(f"Backtest progress: {i+1:,}/{n:,} bars ({current_progress_percent:.1f}%) | {len(trades):,} closed trades", flush=True)
                while next_progress_percent <= current_progress_percent:
                    next_progress_percent += 5.0
                next_progress = time.monotonic() + progress_interval
    if position is not None:
        exit_pos(n - 1, float(series["close"][n - 1]), "end_of_data")
        eq[-1], pos_curve[-1] = cash, 0
    equity = pd.DataFrame({"timestamp": data.timestamp.to_numpy(), "equity": eq, "position": pos_curve})
    for plot_number, values in sorted(plot_values.items()):
        equity[f"plot_{plot_number}"] = values
    reported_inputs = {
        name: value[1].title() if isinstance(value, tuple) and len(value) == 2 and value[0] == "__series_default__" else value
        for name, value in inputs.items()
    }
    if num_contracts is not None:
        if "numcontracts" in reported_inputs:
            reported_inputs["numcontracts"] = num_contracts
        reported_inputs["NumContracts (runner override)"] = num_contracts
    return pd.DataFrame(trades), equity, reported_inputs
