"""Role-restricted Discord interface for the local PowerLanguage backtester."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from run_backtest import ROOT, infer_strategy_settings


load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("easybacktest-discord")


def parse_id_list(value: str, setting_name: str) -> set[int]:
    ids: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdecimal():
            raise ValueError(f"{setting_name} must be comma-separated numeric role IDs")
        ids.add(int(part))
    return ids


TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
ALLOWED_ROLE_IDS = parse_id_list(os.getenv("DISCORD_ALLOWED_ROLE_IDS", ""), "DISCORD_ALLOWED_ROLE_IDS")
MODERATOR_ROLE_IDS = parse_id_list(os.getenv("DISCORD_MODERATOR_ROLE_IDS", ""), "DISCORD_MODERATOR_ROLE_IDS")
def parse_guild_channel_map(value: str) -> dict[int, int]:
    """Read per-server moderator log channels as `guild_id:channel_id` pairs."""
    result: dict[int, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        guild_text, separator, channel_text = part.partition(":")
        if not separator or not guild_text.isdecimal() or not channel_text.isdecimal():
            raise ValueError("DISCORD_MODERATOR_CHANNEL_IDS must be comma-separated guild_id:channel_id pairs")
        result[int(guild_text)] = int(channel_text)
    return result


GUILD_IDS_TEXT = os.getenv("DISCORD_GUILD_IDS", "").strip()
if not GUILD_IDS_TEXT:
    GUILD_IDS_TEXT = os.getenv("DISCORD_GUILD_ID", "").strip()
GUILD_IDS = parse_id_list(GUILD_IDS_TEXT, "DISCORD_GUILD_IDS")
MODERATOR_CHANNEL_IDS_BY_GUILD = parse_guild_channel_map(
    os.getenv("DISCORD_MODERATOR_CHANNEL_IDS", "")
)
# Backward compatibility for the original one-server configuration.
LEGACY_MODERATOR_CHANNEL_TEXT = os.getenv("DISCORD_MODERATOR_CHANNEL_ID", "").strip()
LEGACY_GUILD_TEXT = os.getenv("DISCORD_GUILD_ID", "").strip()
if LEGACY_MODERATOR_CHANNEL_TEXT.isdecimal() and LEGACY_GUILD_TEXT.isdecimal():
    MODERATOR_CHANNEL_IDS_BY_GUILD.setdefault(int(LEGACY_GUILD_TEXT), int(LEGACY_MODERATOR_CHANNEL_TEXT))
MAX_CONCURRENT_BACKTESTS = 3
BACKTEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_BACKTESTS)
ENGINE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_BACKTESTS)
STRATEGIES_DIR = ROOT / "QQQSStrategies"
ACTIVE_BACKTESTS: dict[str, "ActiveBacktest"] = {}


@dataclass
class ActiveBacktest:
    owner_id: int
    status_message: discord.Message
    run_id: str
    strategy_name: str
    moderator_message: discord.Message | None = None
    processes: dict[int, asyncio.subprocess.Process] = field(default_factory=dict)
    stop_requested: bool = False


@asynccontextmanager
async def backtest_slot(run_key: str):
    """Limit engine work and remove queued/running jobs even if cancelled."""
    try:
        async with BACKTEST_SEMAPHORE:
            yield
    finally:
        ACTIVE_BACKTESTS.pop(run_key, None)


class BacktestBot(commands.Bot):
    async def setup_hook(self) -> None:
        if GUILD_IDS:
            for guild_id in sorted(GUILD_IDS):
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                log.info("Registered slash commands in guild %s", guild_id)
        else:
            await self.tree.sync()
            log.info("Registered global slash commands")


bot = BacktestBot(command_prefix=commands.when_mentioned, intents=discord.Intents.default())


def authorized(interaction: discord.Interaction) -> bool:
    if not ALLOWED_ROLE_IDS or not isinstance(interaction.user, discord.Member):
        return False
    return any(role.id in ALLOWED_ROLE_IDS for role in interaction.user.roles)


def has_moderator_role(role_ids: set[int]) -> bool:
    return bool(role_ids & MODERATOR_ROLE_IDS)


def moderator_authorized(interaction: discord.Interaction) -> bool:
    if not MODERATOR_ROLE_IDS or not isinstance(interaction.user, discord.Member):
        return False
    return has_moderator_role({role.id for role in interaction.user.roles})


def can_stop_backtest(user_id: int, role_ids: set[int], active: ActiveBacktest | None) -> bool:
    return active is not None and (user_id == active.owner_id or bool(role_ids & MODERATOR_ROLE_IDS))


async def get_moderator_channel(guild: discord.Guild | None) -> discord.TextChannel | None:
    """Resolve and privacy-check the moderator log channel before posting command data."""
    if guild is None:
        return None
    channel_id = MODERATOR_CHANNEL_IDS_BY_GUILD.get(guild.id)
    if channel_id is None:
        return None
    channel = bot.get_channel(channel_id)
    if channel is None:
        try:
            channel = await bot.fetch_channel(channel_id)
        except (discord.HTTPException, discord.NotFound, discord.Forbidden) as exc:
            log.error("Cannot access configured moderator channel: %s", type(exc).__name__)
            return None
    if not isinstance(channel, discord.TextChannel) or channel.guild.id != guild.id:
        log.error("Configured moderator channel is not a text channel in this server")
        return None
    if channel.permissions_for(guild.default_role).view_channel:
        log.error("Moderator channel is visible to @everyone; refusing to copy command data there")
        return None
    moderator_roles = [guild.get_role(role_id) for role_id in MODERATOR_ROLE_IDS]
    moderator_roles = [role for role in moderator_roles if role is not None]
    if not moderator_roles or not any(channel.permissions_for(role).view_channel for role in moderator_roles):
        log.error("Configured moderator role cannot view the moderator channel")
        return None
    for role in guild.roles:
        if role.id in MODERATOR_ROLE_IDS or role.id == guild.default_role.id or role.permissions.administrator:
            continue
        if channel.permissions_for(role).view_channel:
            log.error("A non-moderator role can view the moderator channel; refusing to copy command data")
            return None
    return channel


async def copy_to_moderator_channel(
    guild: discord.Guild | None,
    content: str,
    file_paths: list[Path] | None = None,
) -> discord.Message | None:
    channel = await get_moderator_channel(guild)
    if channel is None:
        return None
    files = [discord.File(path, filename=path.name) for path in (file_paths or []) if path.is_file()]
    try:
        return await channel.send(content=content, files=files, allowed_mentions=discord.AllowedMentions.none())
    except (discord.HTTPException, discord.Forbidden) as exc:
        log.error("Could not copy command response to moderator channel: %s", type(exc).__name__)
        return None
    finally:
        for file in files:
            file.close()


async def private_reply(
    interaction: discord.Interaction,
    content: str,
    file_path: Path | None = None,
) -> None:
    """Reply privately to the caller and copy the same outcome into mod-only logs."""
    if file_path is not None and file_path.is_file():
        await interaction.response.send_message(content, file=discord.File(file_path), ephemeral=True)
    else:
        await interaction.response.send_message(content, ephemeral=True)
    command_name = interaction.command.name if interaction.command else "command"
    moderator_content = f"**/{command_name} · {interaction.user}**\n{content}"
    await copy_to_moderator_channel(interaction.guild, moderator_content, [file_path] if file_path else None)


async def edit_status_with_moderator_copy(job: ActiveBacktest, content: str) -> None:
    try:
        await job.status_message.edit(content=content)
    except discord.HTTPException as exc:
        log.error("Could not update caller backtest status: %s", type(exc).__name__)
    if job.moderator_message is not None:
        try:
            await job.moderator_message.edit(content=content)
        except discord.HTTPException as exc:
            log.error("Could not update moderator backtest status: %s", type(exc).__name__)


def uploaded_strategy_filename(attachment: discord.Attachment) -> str:
    basename = attachment.filename.replace("\\", "/").rsplit("/", 1)[-1]
    if Path(basename).suffix.lower() != ".pl":
        raise ValueError("Attach a PowerLanguage script with a `.pl` extension.")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(basename).stem).strip("._") or "uploaded_strategy"
    return f"{stem}.pl"


def format_duration(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def parse_backtest_matrix(asset_text: str, timeframe_text: str, contract_text: str) -> list[tuple[str, str, float]]:
    allowed_assets = {"NQ", "ES", "GC", "BTC", "MNQ", "MES", "MGC", "CL", "ZN"}
    assets = list(dict.fromkeys(part.strip().upper() for part in asset_text.split(",") if part.strip()))
    bad_assets = [item for item in assets if item not in allowed_assets]
    if not assets or bad_assets:
        raise ValueError("Assets must be comma-separated supported symbols: NQ, ES, GC, BTC, MNQ, MES, MGC, CL, ZN.")

    timeframes = list(dict.fromkeys(part.strip() for part in timeframe_text.split(",") if part.strip()))
    if not timeframes:
        raise ValueError("Enter one or more comma-separated timeframes, such as 1min, 5min, 1h.")
    normalized_timeframes: list[str] = []
    for item in timeframes:
        if item.lower() == "auto":
            normalized_timeframes.append("auto")
            continue
        if not re.fullmatch(r"\d+\s*(?:min|mins|minute|minutes|m|h|hr|hrs|hour|hours|d|day|days|w|week|weeks)", item, re.I):
            raise ValueError(f"Invalid timeframe {item!r}; use values such as 1min, 5min, 1h, 4h, or 1d.")
        normalized_timeframes.append(item.replace(" ", ""))
    if "auto" in (value.lower() for value in normalized_timeframes) and len(normalized_timeframes) > 1:
        raise ValueError("Use `auto` by itself or provide explicit comma-separated timeframes.")

    sizes: list[float] = []
    for item in contract_text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid contract size {item!r}; use positive numbers such as 1, 0.5, or 0.1.") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Contract sizes must be positive finite numbers.")
        if value not in sizes:
            sizes.append(value)
    if not sizes:
        raise ValueError("Enter one or more comma-separated contract sizes, such as 1, 0.5, 0.1.")

    combinations = [(asset, timeframe, size) for asset in assets for timeframe in normalized_timeframes for size in sizes]
    if len(combinations) > 36:
        raise ValueError(f"That selection creates {len(combinations)} backtests; the maximum per command is 36 combinations.")
    return combinations


def terminal_progress_line(line: str, asset: str, width: int = 24) -> str | None:
    """Render engine progress as an ASCII bar in the bot's Windows console."""
    phase = None
    current = total = None
    detail = ""
    percent = None
    match = re.search(r"Backtest progress:\s*([\d,]+)/([\d,]+) bars\s*\(([\d.]+)%\)\s*\|\s*([\d,]+) closed trades", line)
    if match:
        phase = "Backtest"
        current, total = int(match.group(1).replace(",", "")), int(match.group(2).replace(",", ""))
        percent = float(match.group(3))
        detail = f"{current:,}/{total:,} bars | {match.group(4)} closed trades"
    else:
        match = re.search(r"Monte Carlo progress:\s*([\d,]+) simulated equity curves\s*\(([\d.]+)%\)", line)
        if match:
            phase = "Monte Carlo"
            current = int(match.group(1).replace(",", ""))
            percent = float(match.group(2))
            total = max(current, round(current * 100 / percent)) if percent > 0 else current
            detail = f"{current:,}/{total:,} curves"
        else:
            match = re.search(r"Crypto prop progress:\s*([\d.]+)%\s*\(([\d,]+) account tier\)", line)
            if match:
                phase = "Crypto prop"
                percent = float(match.group(1))
                detail = f"account tier {match.group(2)}"
            else:
                match = re.search(r"(?:(?:TBBO conversion|Order-event conversion):\s*|book conversion: event file\s+)(\d+)/([\d,]+)", line)
                if match:
                    phase = "Data conversion"
                    current = int(match.group(1).replace(",", ""))
                    total = int(match.group(2).replace(",", ""))
                    percent = current * 100.0 / total if total else 0.0
                    detail = f"{current:,}/{total:,} files"
    if phase is None or percent is None:
        return None
    percent = min(100.0, max(0.0, percent))
    filled = min(width, max(0, int(round(width * percent / 100))))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{asset}] {phase:<13} [{bar}] {percent:5.1f}%  {detail}"


async def save_uploaded_strategy(attachment: discord.Attachment, filename: str) -> Path:
    """Persist an uploaded script without replacing a previously received file."""
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(filename)
    suffix = datetime.now(timezone.utc).strftime("_%Y%m%d_%H%M%S_%f")
    candidates = [STRATEGIES_DIR / filename]
    candidates.extend(
        STRATEGIES_DIR / f"{base.stem}{suffix}_{index}{base.suffix}"
        for index in range(1000)
    )
    for candidate in candidates:
        try:
            destination = candidate.open("xb")
        except FileExistsError:
            continue
        try:
            await attachment.save(destination)
            return candidate
        except Exception:
            destination.close()
            candidate.unlink(missing_ok=True)
            raise
        finally:
            if not destination.closed:
                destination.close()
    raise FileExistsError("Could not create a unique saved filename in QQQSStrategies.")


async def save_uploaded_data2(attachment: discord.Attachment, run_id: str) -> Path:
    """Save a secondary OHLCV feed temporarily for this one backtest."""
    tmp_dir = ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lowered = attachment.filename.lower()
    suffix = ".csv.gz" if lowered.endswith(".csv.gz") else Path(lowered).suffix
    destination = tmp_dir / f"discord-data2-{run_id}{suffix}"
    with destination.open("xb") as stream:
        try:
            await attachment.save(stream)
        except Exception:
            stream.close()
            destination.unlink(missing_ok=True)
            raise
    return destination


def parse_result(stdout: str) -> dict:
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{\s*"asset"\s*:', stdout):
        try:
            result, _ = decoder.raw_decode(stdout[match.start():])
            return result
        except json.JSONDecodeError:
            continue
    raise ValueError("The backtest finished without returning its result summary.")


def safe_error(text: str) -> str:
    cleaned = text.replace(str(ROOT), "[engine]")
    cleaned = re.sub(r"[A-Za-z]:\\Users\\[^\\\s]+", "[local path]", cleaned)
    return cleaned[-1400:]


def brief_error(details: str) -> str:
    """Return a short actionable error line without traceback or source output."""
    lines = [line.strip() for line in details.splitlines() if line.strip()]
    message = "The backtest could not be completed."
    for line in reversed(lines):
        if line.startswith(("Traceback (most recent call last):", "File \"", "usage:")):
            continue
        if ": error:" in line:
            message = line.split(": error:", 1)[1].strip()
            break
        if re.match(r"(?:[A-Za-z_][\w.]*Error|Exception):", line):
            message = line.split(":", 1)[1].strip() or line.split(":", 1)[0]
            break
        if not line.startswith(("Backtest progress:", "Monte Carlo progress:")):
            message = line
            break
    message = safe_error(message).replace("`", "'").replace("\n", " ")
    if len(message) > 180:
        message = message[:177].rstrip() + "..."
    return f"An error occurred: {message}"


def make_results_archive(folder: Path, max_part_size: int) -> tuple[Path, list[Path]]:
    temp_dir = Path(tempfile.mkdtemp(prefix="easybacktest-discord-"))
    archive_path = temp_dir / f"{folder.parent.name}_{folder.name}_results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            archive.write(path, arcname=path.name)
    size = archive_path.stat().st_size
    if size <= max_part_size:
        return temp_dir, [archive_path]
    parts: list[Path] = []
    chunk_size = max(1, max_part_size - 256 * 1024)
    with archive_path.open("rb") as source:
        index = 1
        while chunk := source.read(chunk_size):
            part = temp_dir / f"{archive_path.stem}.zip.part{index:03d}"
            part.write_bytes(chunk)
            parts.append(part)
            index += 1
    archive_path.unlink()
    return temp_dir, parts


@bot.tree.command(name="backtest", description="Backtest the attached PowerLanguage script and post results here")
@app_commands.choices(
    lucidflex_account=[app_commands.Choice(name=x, value=x) for x in ("25k", "50k", "100k", "150k")],
    lucidpro_account=[app_commands.Choice(name=x, value=x) for x in ("25k", "50k", "100k", "150k")],
    lucid_pnl_basis=[app_commands.Choice(name="Point movement (historical contract P&L)", value="points"),
                     app_commands.Choice(name="Scale to current notional", value="current_notional")],
)
@app_commands.describe(
    asset="Comma-separated assets, such as NQ,ES",
    strategy_file="Attach the .pl script to run; local strategies are not used",
    data2_file="Optional separate Data2 OHLCV feed (CSV, CSV.GZ, or Parquet)",
    data_period="History window, such as 90d, 4y, or all",
    timeframe="Comma-separated bar intervals, such as 1min,5min, or auto",
    num_contracts="Comma-separated sizes; fractional sizes such as 1,0.5,0.1 are allowed",
    lucidpro_dll="Apply LucidPro's optional daily loss limit",
    lucid_pnl_basis="Use historical point P&L or scale trade returns to current notional",
)
async def backtest(
    interaction: discord.Interaction,
    asset: str,
    strategy_file: discord.Attachment,
    data2_file: discord.Attachment | None = None,
    data_period: str = "all",
    timeframe: str = "auto",
    num_contracts: str = "1",
    lucidflex_account: str = "100k",
    lucidpro_account: str = "100k",
    lucidpro_dll: bool = False,
    lucid_pnl_basis: str = "points",
) -> None:
    if not authorized(interaction):
        await private_reply(interaction,
            "You don't have an allowed role for backtests. The bot owner must configure `DISCORD_ALLOWED_ROLE_IDS`.",
        )
        return
    try:
        combinations = parse_backtest_matrix(asset, timeframe, num_contracts)
    except ValueError as exc:
        await private_reply(interaction, str(exc))
        return
    assets = list(dict.fromkeys(item[0] for item in combinations))
    timeframes = list(dict.fromkeys(item[1] for item in combinations))
    contract_sizes = list(dict.fromkeys(item[2] for item in combinations))
    try:
        filename = uploaded_strategy_filename(strategy_file)
    except ValueError as exc:
        await private_reply(interaction, str(exc))
        return
    if strategy_file.size <= 0 or strategy_file.size > 2 * 1024 * 1024:
        await private_reply(interaction, "The attached .pl file must be no larger than 2 MiB.")
        return
    if data2_file is not None:
        data2_name = data2_file.filename.lower()
        if not data2_name.endswith((".csv", ".csv.gz", ".parquet")):
            await private_reply(interaction, "Data2 must be a CSV, CSV.GZ, or Parquet OHLCV file.")
            return
        if data2_file.size <= 0 or data2_file.size > 100 * 1024 * 1024:
            await private_reply(interaction, "The optional Data2 file must be no larger than 100 MiB.")
            return

    try:
        await interaction.response.defer(ephemeral=True, thinking=True)
    except discord.InteractionResponded:
        return
    if interaction.channel is None:
        await interaction.followup.send("Run this command in a server channel.", ephemeral=True)
        await copy_to_moderator_channel(interaction.guild, f"**/backtest · {interaction.user}**\nRun this command in a server channel.")
        return

    run_id = str(interaction.id)
    tag = datetime.now(timezone.utc).strftime("discord-%Y%m%d-%H%M%S") + f"-{interaction.id}"
    queued = BACKTEST_SEMAPHORE.locked()
    queue_notice = "Queued; three backtests are currently using the engine."
    status = await interaction.channel.send(
        (f"Run `{run_id}`: queued for **{len(combinations)} backtest combinations**; the engine is at its three-combination concurrency limit."
         if queued else
         f"Run `{run_id}`: preparing **{filename}** for **{len(combinations)} combinations**.\n"
         f"Assets: {', '.join(assets)} · Timeframes: {', '.join(timeframes)} · Sizes: {', '.join(f'{x:g}' for x in contract_sizes)} contracts.\n"
         "Runtime estimate will appear after the first combination loads data.")
    )
    moderator_status = await copy_to_moderator_channel(
        interaction.guild,
        f"**/backtest started by {interaction.user} · run `{run_id}`**\n"
        + (queue_notice if queued else
           f"Preparing **{filename}** for **{len(combinations)} combinations**.\n"
           f"Assets: {', '.join(assets)} · Timeframes: {', '.join(timeframes)} · Sizes: {', '.join(f'{x:g}' for x in contract_sizes)} contracts.\n"
           "Runtime estimate will appear after the first combination loads data."),
    )
    if queued:
        try:
            await interaction.followup.send(queue_notice, ephemeral=True)
        except discord.HTTPException as exc:
            log.warning("Could not send backtest queue notice: %s", type(exc).__name__)
    active_job = ActiveBacktest(owner_id=interaction.user.id, status_message=status,
                                run_id=run_id, strategy_name=Path(filename).stem,
                                moderator_message=moderator_status)
    ACTIVE_BACKTESTS[str(interaction.id)] = active_job
    async with backtest_slot(str(interaction.id)):
        data2_path: Path | None = None
        try:
            if active_job.stop_requested:
                await edit_status_with_moderator_copy(active_job, f"Run `{run_id}` cancelled before it started.")
                return
            script = await save_uploaded_strategy(strategy_file, filename)
            source_text = script.read_text(encoding="utf-8-sig")
            external_data2_required = bool(re.search(
                r"(?im)^\s*//\s*ENGINE_DATA2_EXTERNAL_REQUIRED\s*=\s*true\s*$", source_text
            ))
            if external_data2_required and data2_file is None:
                raise ValueError("This strategy needs a separate Data2 implied-volatility file. Attach it in the Data2 field.")
            if data2_file is not None:
                data2_path = await save_uploaded_data2(data2_file, run_id)
            settings = infer_strategy_settings(source_text)
            batch_started_at = time.monotonic()
            failures: list[str] = []
            async def run_combination(combo_index: int, asset: str, requested_timeframe: str, num_contracts: float):
                async with ENGINE_SEMAPHORE:
                    stdout_chunks = []
                    stderr_lines = []
                    process_started_at = 0.0
                    runtime_state = {
                        "engine_started_at": None,
                        "estimated_total_seconds": None,
                        "estimated_runtime_basis": "",
                    }
                    interval = settings.get("bar_timeframe", "1min") if requested_timeframe.lower() == "auto" else requested_timeframe
                    combo_tag = tag + "-" + re.sub(r"[^A-Za-z0-9._-]+", "_", f"{asset}-{interval}-{num_contracts:g}-{combo_index}")
                    timezone_name = settings.get("session_timezone")
                    command = [
                        sys.executable, str(ROOT / "run_backtest.py"),
                        "--asset", asset,
                        "--strategy", str(script),
                        "--data-period", data_period.strip() or "all",
                        "--bar-timeframe", interval,
                        "--num-contracts", f"{num_contracts:g}",
                        "--lucidflex-account", lucidflex_account,
                        "--lucidpro-account", lucidpro_account,
                        "--lucidpro-dll", "on" if lucidpro_dll else "off",
                        "--lucid-pnl-basis", lucid_pnl_basis,
                        "--run-tag", combo_tag,
                    ]
                    if data2_path is not None:
                        command.extend(["--data2", str(data2_path)])
                    if timezone_name:
                        command.extend(["--session-timezone", timezone_name])
                    if active_job.stop_requested:
                        await edit_status_with_moderator_copy(active_job, "Backtest stopped.")
                        return
                    await edit_status_with_moderator_copy(
                        active_job,
                        f"Run `{run_id}`: starting **{script.stem}** on **{asset}** ({data_period}, {interval}, {num_contracts:g} contracts).",
                    )
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=ROOT,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    active_job.processes[combo_index] = process
                    process_started_at = time.monotonic()
                    if active_job.stop_requested and process.returncode is None:
                        process.terminate()
                    assert process.stdout is not None and process.stderr is not None

                    async def consume_stdout() -> None:
                        while raw_line := await process.stdout.readline():
                            stdout_chunks.append(raw_line)
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if not line:
                                continue
                            terminal_line = terminal_progress_line(line, asset)
                            if terminal_line is not None:
                                print(terminal_line, flush=True)
                                progress_match = re.search(r"Backtest progress:.*?\(([\d.]+)%\)", line)
                                engine_started_at = runtime_state["engine_started_at"]
                                if progress_match and isinstance(engine_started_at, (float, int)):
                                    percent = float(progress_match.group(1))
                                    engine_elapsed = time.monotonic() - engine_started_at
                                    if engine_elapsed >= 8 and percent >= 10:
                                        prep_elapsed = max(0.0, engine_started_at - process_started_at)
                                        old_estimate = runtime_state["estimated_total_seconds"]
                                        postprocess_pad = max(15.0, (float(old_estimate or 0.0)) * 0.08)
                                        runtime_state["estimated_total_seconds"] = (
                                            prep_elapsed + engine_elapsed * 100.0 / percent + postprocess_pad
                                        )

                    async def consume_stderr() -> None:
                        while raw_line := await process.stderr.readline():
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if not line:
                                continue
                            stderr_lines.append(line)
                            del stderr_lines[:-80]
                            print(terminal_progress_line(line, asset) or f"[engine] {line}", flush=True)
                            if "strategy bars at" in line and "Starting..." in line and runtime_state["engine_started_at"] is None:
                                runtime_state["engine_started_at"] = time.monotonic()
                            estimate_match = re.search(
                                r"Runtime estimate: total_seconds=(\d+(?:\.\d+)?) remaining_seconds=(\d+(?:\.\d+)?); basis=(.+)$",
                                line,
                            )
                            if estimate_match:
                                runtime_state["estimated_total_seconds"] = float(estimate_match.group(1))
                                runtime_state["estimated_runtime_basis"] = estimate_match.group(3)
                                batch_elapsed = time.monotonic() - batch_started_at
                                remaining_combinations = max(0, len(combinations) - combo_index + 1)
                                remaining_waves = math.ceil(remaining_combinations / MAX_CONCURRENT_BACKTESTS)
                                estimated_batch_seconds = batch_elapsed + (
                                    float(runtime_state["estimated_total_seconds"]) * remaining_waves
                                )
                                await edit_status_with_moderator_copy(
                                    active_job,
                                    f"**Combination {combo_index}/{len(combinations)} · {script.stem} · {asset}**\n"
                                    f"Estimated batch time: **about {format_duration(estimated_batch_seconds)}** "
                                    f"({format_duration(float(estimate_match.group(2)))} remaining in this combination after data preparation).\n"
                                    f"Estimate basis: {runtime_state['estimated_runtime_basis']}.",
                                )

                    await asyncio.gather(process.wait(), consume_stderr(), consume_stdout())
                    active_job.processes.pop(combo_index, None)
                    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
                    if process.returncode != 0:
                        if active_job.stop_requested:
                            await edit_status_with_moderator_copy(active_job, "Backtest stopped.")
                        else:
                            details = "\n".join(stderr_lines) or stdout or f"process exited with code {process.returncode}"
                            log.error("Backtest subprocess failed (exit %s):\n%s", process.returncode, details)
                            failure = f"{asset} · {interval} · {num_contracts:g} contracts: {brief_error(details)}"
                            failures.append(failure)
                            await edit_status_with_moderator_copy(active_job, f"Combination {combo_index}/{len(combinations)} failed. {brief_error(details)}")
                        return

                    result = parse_result(stdout)
                    result_folder = Path(result["result_folder"]).resolve()
                    strategy_folder = re.sub(r"[^A-Za-z0-9._-]+", "_", script.stem).strip("._") or "strategy"
                    expected_root = (ROOT / "results" / asset / strategy_folder / combo_tag).resolve()
                    if result_folder != expected_root or not result_folder.is_dir():
                        raise ValueError("The runner returned an unexpected or missing results folder.")
                    stats = result.get("stats", {})
                    net_profit = float(stats.get("net_profit", 0.0))
                    status_text = (
                        f"**Combination {combo_index}/{len(combinations)} complete · {asset} · {script.stem} · run `{run_id}`**\n"
                        f"Net P&L: **${net_profit:,.3f}** · Trades: **{int(stats.get('trade_count', 0))}** · "
                        f"Max drawdown: **${float(stats.get('max_drawdown', 0.0)):,.3f}**\n"
                        f"Timeframe: {interval} · Data: {data_period} · Size: {num_contracts:g} contracts"
                    )
                    actual_runtime_seconds = time.monotonic() - process_started_at if process_started_at else 0.0
                    estimated_total_seconds = runtime_state["estimated_total_seconds"]
                    if isinstance(estimated_total_seconds, (float, int)):
                        status_text += (
                            f"\nEstimated runtime: {format_duration(float(estimated_total_seconds))} · "
                            f"Actual engine runtime: {format_duration(actual_runtime_seconds)}"
                        )
                    else:
                        status_text += f"\nActual engine runtime: {format_duration(actual_runtime_seconds)}"
                    if asset == "BTC" and isinstance(stats.get("crypto_prop"), dict):
                        account_50k = stats["crypto_prop"].get("accounts", {}).get("50000", {})
                        status_text += (
                            f"\nCrypto prop average · 50k: pass **{float(account_50k.get('evaluation_pass_rate_pct', 0)):.3f}%**"
                            f" · breach **{float(account_50k.get('drawdown_breach_rate_pct', 0)):.3f}%**"
                        )
                    await edit_status_with_moderator_copy(active_job, status_text)

                    upload_limit = int(getattr(interaction.guild, "filesize_limit", 25 * 1024 * 1024))
                    temp_dir, parts = await asyncio.to_thread(make_results_archive, result_folder, upload_limit)
                    try:
                        previews = ["equity_curve.png", "strategy_plots.png", "monte_carlo_equity.png", "monte_carlo.png", "report.pdf"]
                        for name in previews:
                            path = result_folder / name
                            if path.is_file() and path.stat().st_size <= upload_limit:
                                await interaction.channel.send(file=discord.File(path, filename=name))
                                await copy_to_moderator_channel(
                                    interaction.guild,
                                    f"**/backtest result · {asset} · {name}**",
                                    [path],
                                )
                        for offset in range(0, len(parts), 10):
                            batch = parts[offset:offset + 10]
                            files = [discord.File(path, filename=path.name) for path in batch]
                            note = "Full results archive attached. It contains the report, stats, trades, equity data, and charts."
                            if len(parts) > 1:
                                note = (f"Results archive part {offset + 1}–{offset + len(batch)} of {len(parts)} attached. "
                                        "If split into parts, concatenate them in numeric order before opening the ZIP.")
                            await interaction.channel.send(content=note, files=files)
                            await copy_to_moderator_channel(
                                interaction.guild,
                                f"**/backtest result · {asset}**\n{note}",
                                batch,
                            )
                    finally:
                        for path in temp_dir.iterdir():
                            path.unlink(missing_ok=True)
                        temp_dir.rmdir()
                    return {
                        "asset": asset,
                        "timeframe": interval,
                        "contracts": num_contracts,
                        "net_profit": net_profit,
                        "trade_count": int(stats.get("trade_count", 0)),
                        "max_drawdown": float(stats.get("max_drawdown", 0.0)),
                    }
            combo_tasks = [
                asyncio.create_task(run_combination(index, asset_name, timeframe_name, size))
                for index, (asset_name, timeframe_name, size) in enumerate(combinations, 1)
            ]
            try:
                combo_results = await asyncio.gather(*combo_tasks)
            except BaseException:
                active_job.stop_requested = True
                for child in list(active_job.processes.values()):
                    if child.returncode is None:
                        child.terminate()
                await asyncio.gather(*combo_tasks, return_exceptions=True)
                raise
            combo_summaries = [result for result in combo_results if result is not None]
            if active_job.stop_requested:
                await edit_status_with_moderator_copy(active_job, "Backtest stopped.")
                return
            summary_lines = [
                f"{item['asset']} · {item['timeframe']} · {item['contracts']:g} contracts: "
                f"P&L ${item['net_profit']:,.3f}, {item['trade_count']} trades, "
                f"max DD ${item['max_drawdown']:,.3f}"
                for item in combo_summaries
            ]
            summary_lines.extend(f"FAILED · {failure}" for failure in failures)
            await edit_status_with_moderator_copy(
                active_job,
                f"**Backtest batch finished · {script.stem} · run `{run_id}`**\n"
                f"Succeeded: {len(combo_summaries)}/{len(combinations)} · Failed: {len(failures)}. Detailed summary follows.",
            )
            for offset in range(0, len(summary_lines), 12):
                batch_lines = summary_lines[offset:offset + 12]
                summary_message = (
                    f"**Batch summary · {script.stem} · {offset + 1}–{offset + len(batch_lines)} "
                    f"of {len(summary_lines)}**\n" + "\n".join(batch_lines)
                )
                await interaction.channel.send(summary_message)
                await copy_to_moderator_channel(interaction.guild, summary_message)

        except asyncio.CancelledError:
            for child in list(active_job.processes.values()):
                if child.returncode is None:
                    child.terminate()
            await asyncio.gather(*(child.wait() for child in list(active_job.processes.values())),
                                 return_exceptions=True)
            raise
        except Exception as exc:
            log.exception("Discord backtest handler failed")
            try:
                await edit_status_with_moderator_copy(active_job, brief_error(str(exc)))
            except discord.HTTPException:
                pass
        finally:
            if data2_path is not None:
                data2_path.unlink(missing_ok=True)


@bot.tree.command(name="stop", description="Stop one of your active backtests; moderators can stop any active run")
@app_commands.describe(run_id="The run ID shown in the backtest status; omit when only one run is active")
async def stop_backtest(interaction: discord.Interaction, run_id: str = "") -> None:
    role_ids = {role.id for role in interaction.user.roles} if isinstance(interaction.user, discord.Member) else set()
    active_jobs = list(ACTIVE_BACKTESTS.values())
    if not active_jobs:
        await private_reply(interaction, "There is no active backtest to stop.")
        return
    eligible_jobs = [job for job in active_jobs if can_stop_backtest(interaction.user.id, role_ids, job)]
    if not eligible_jobs:
        await private_reply(interaction,
            "Only the person who started this backtest or a configured moderator can stop it.",
        )
        return
    requested_id = run_id.strip()
    if requested_id:
        matches = [job for job in eligible_jobs if job.run_id == requested_id]
        if len(matches) != 1:
            visible = ", ".join(f"`{job.run_id}` ({job.strategy_name})" for job in eligible_jobs)
            await private_reply(interaction, f"No active run matches `{requested_id}`. Available runs: {visible}.")
            return
        job = matches[0]
    elif len(eligible_jobs) > 1:
        visible = ", ".join(f"`{job.run_id}` ({job.strategy_name})" for job in eligible_jobs)
        await private_reply(interaction, f"More than one run is active. Choose a run ID: {visible}.")
        return
    else:
        job = eligible_jobs[0]
    job.stop_requested = True
    if job.processes:
        for child in list(job.processes.values()):
            if child.returncode is not None:
                continue
            try:
                child.terminate()
            except ProcessLookupError:
                pass
    await private_reply(interaction, f"Stop requested for run `{job.run_id}` ({job.strategy_name}).")


@bot.tree.command(name="guide", description="Show the backtesting bot commands and their options")
async def guide(interaction: discord.Interaction) -> None:
    if not authorized(interaction):
        await private_reply(interaction, "You don't have an allowed role for this bot.")
        return
    text = (
        "**Backtest bot commands**\n"
        "`/backtest` - attach a `.pl` file; enter comma-separated assets, timeframes, and contract sizes to run their combinations in one batch (up to 36). Example: assets `NQ,ES`, timeframes `1min,5min`, sizes `1,0.5` runs 8 tests.\n"
        "Each combination has its own result folder and report. Up to 3 individual combinations run simultaneously across all requests.\n"
        "BTC runs an averaged crypto-prop evaluation model for 5k, 10k, 25k, 50k, and 100k tiers.\n"
        "`/stop` - stop your active run (include its run ID if you have several); moderators can stop any run.\n"
        "`/datasets` - see which local OHLCV and quote/order-flow feeds are available.\n"
        "Command replies and backtest progress/results are also copied to the private moderator channel.\n"
        "`/guide` - show this help."
    )
    if moderator_authorized(interaction):
        text += "\n`/latest` - show the newest completed backtest and its PDF report."
    await private_reply(interaction, text)


@bot.tree.command(name="datasets", description="Show local datasets available to the backtesting engine")
async def datasets(interaction: discord.Interaction) -> None:
    if not authorized(interaction):
        await private_reply(interaction, "You don't have an allowed role for this bot.")
        return
    asset_names = ("NQ", "ES", "GC", "BTC", "MNQ", "MES", "MGC", "CL", "ZN")
    lines = ["**Local backtest data**"]
    for asset in asset_names:
        candidates = [ROOT / "data" / f"{asset}.csv", ROOT / "data" / f"{asset}.parquet",
                      ROOT / "data" / f"{asset}.csv.gz",
                      ROOT / "data" / asset / "ohlcv" / f"{asset.lower()}_1m_full.parquet"]
        available = [path.name if path.parent == ROOT / "data" else f"{asset}/ohlcv/{path.name}"
                     for path in candidates if path.is_file()]
        tbbo = sorted((ROOT / "data" / "raw" / "tbbo").glob(f"{asset}_*_tbbo.dbn"))
        if tbbo:
            available.append(f"TBBO ({len(tbbo)} file{'s' if len(tbbo) != 1 else ''})")
        if asset == "NQ" and (ROOT / "data" / "NQ" / "raw" / "orderbook" / "daily_events").is_dir():
            available.append("order-event feed")
        imported_root = ROOT / "data" / "imported_20261030"
        tbbo_dir = imported_root / "futures_tbbo" / asset
        if tbbo_dir.is_dir():
            available.append(f"imported TBBO ({len(list(tbbo_dir.glob('*.parquet')))} sessions)")
        if asset == "NQ" and (imported_root / "l2_book" / "NQ_MBP10" / "NQ_MBP10_5s_clean.parquet").exists():
            available.append("imported MBP-10 summaries (63 sessions; timestamps absent, not synchronized)")
        if asset == "MGC" and (imported_root / "l2_book" / "MGC" / "MGC_BBO.parquet").exists():
            available.append("imported MGC order book (5 days; features joined to trade bars)")
        lines.append(f"**{asset}:** {', '.join(available) if available else 'no local file found'}")
    await private_reply(interaction, "\n".join(lines))


@bot.tree.command(name="latest", description="Show the newest completed backtest and its PDF report")
async def latest(interaction: discord.Interaction) -> None:
    if not moderator_authorized(interaction):
        await private_reply(interaction, "This command is available to moderators only.")
        return
    result_files = list((ROOT / "results").glob("*/*/stats.json"))
    result_files.extend((ROOT / "results").glob("*/*/*/stats.json"))
    if not result_files:
        await private_reply(interaction, "No completed backtest results were found.")
        return
    stats_path = max(result_files, key=lambda path: path.stat().st_mtime)
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        await private_reply(interaction, "The latest result summary could not be read.")
        return
    result_parts = stats_path.relative_to(ROOT / "results").parts
    asset = result_parts[0]
    content = (
        f"**Latest backtest · {asset}**\n"
        f"Net P&L: **${float(stats.get('net_profit', 0)):,.3f}** · "
        f"Trades: **{int(stats.get('trade_count', 0))}** · "
        f"Max drawdown: **${float(stats.get('max_drawdown', 0)):,.3f}**"
    )
    pdf_path = stats_path.parent / "report.pdf"
    if pdf_path.is_file():
        await private_reply(interaction, content, pdf_path)
    else:
        await private_reply(interaction, content + "\nPDF report is missing from this result folder.")


@bot.event
async def on_ready() -> None:
    log.info("Connected as %s", bot.user)
    if not ALLOWED_ROLE_IDS:
        log.warning("No allowed role IDs configured; /backtest is locked until DISCORD_ALLOWED_ROLE_IDS is set")
    if not MODERATOR_ROLE_IDS:
        log.warning("No moderator role IDs configured; only the backtest starter can use /stop")
    for guild_id in sorted(GUILD_IDS):
        guild = bot.get_guild(guild_id)
        if guild is None:
            log.warning("Bot is not installed in configured guild %s", guild_id)
        elif guild_id in MODERATOR_CHANNEL_IDS_BY_GUILD:
            channel = await get_moderator_channel(guild)
            if channel is not None:
                log.info("Moderator command copies enabled in guild %s", guild_id)
        else:
            log.info("Moderator-channel logging is disabled in guild %s", guild_id)


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Set DISCORD_BOT_TOKEN in .env before starting the bot.")
    bot.run(TOKEN)
