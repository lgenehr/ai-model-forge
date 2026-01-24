import click
from .pipeline.builder import DatasetBuilder
from .utils.logger import setup_logger

logger = setup_logger("cli")

@click.group()
def cli():
    pass

@cli.command()
@click.option('--year', multiple=True, type=int, default=[2025, 2026], help='Years to include')
@click.option('--out', default='dataset/dataset_2025_2026.jsonl', help='Output file path')
@click.option('--max_items', default=10000, help='Maximum items to generate')
def build(year, out, max_items):
    """Build the fine-tuning dataset."""
    logger.info(f"Building dataset for years {year} to {out}")
    builder = DatasetBuilder()
    builder.build(list(year), out, max_items)

if __name__ == '__main__':
    cli()
