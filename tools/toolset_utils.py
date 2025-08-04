from pathlib import Path

import polars as pl
#from pydantic_ai import RunContext

from .finn_deps import FinnDeps, RunContext


def resolve_df_path(ctx: RunContext, df_path: str | Path) -> Path:
    if isinstance(df_path, str):
        df_path = Path(df_path)
    paths_to_try = [
        ctx.deps.analysis_dir / df_path.name,
        (ctx.deps.analysis_dir / df_path.stem).with_suffix(".parquet"),
        (ctx.deps.analysis_dir / df_path.stem).with_suffix(".csv"),
        ctx.deps.data_dir / df_path.name,
        (ctx.deps.data_dir / df_path.stem).with_suffix(".parquet"),
        (ctx.deps.data_dir / df_path.stem).with_suffix(".csv"),
    ]
    for path in paths_to_try:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"DataFrame file not found in expected locations: {paths_to_try}"
    )


def load_df(ctx: RunContext, data: pl.DataFrame | str | Path) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data
    df_path = resolve_df_path(ctx, data)
    return (
        pl.read_parquet(df_path)
        if df_path.suffix.lower() == ".parquet"
        else pl.read_csv(df_path)
    )


def df_path_to_analysis_df_path(ctx: RunContext, df_path: str | Path) -> Path:
    df_path = Path(df_path)
    analysis_path = ctx.deps.analysis_dir / df_path.name
    if analysis_path.suffix.lower() not in [".parquet", ".csv"]:
        analysis_path = (ctx.deps.analysis_dir / df_path.stem).with_suffix(".parquet")
    return analysis_path


def save_df_to_analysis_dir(
    ctx: RunContext, df: pl.DataFrame, analysis_result_file_name: str
) -> Path:
    output_path = df_path_to_analysis_df_path(ctx, analysis_result_file_name)
    if output_path.suffix.lower() == ".parquet":
        df.write_parquet(output_path)
    else:
        df.write_csv(output_path)
    return output_path
