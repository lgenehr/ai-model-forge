"""
TUI Utils - Utilities for creating rich terminal user interfaces.

This module provides helpers for creating beautiful and informative
terminal interfaces using the Rich library.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.tree import Tree
from rich import box
from rich.style import Style
from datetime import datetime
import time

# Console global para output formatado
console = Console()


class TrainingUI:
    """Gerencia a interface de usuário para treinamento."""

    def __init__(self):
        self.console = console
        self.start_time = None

    def print_header(self, title: str, subtitle: str = ""):
        """Imprime um cabeçalho destacado."""
        header_text = Text(title, style="bold cyan")
        if subtitle:
            header_text.append(f"\n{subtitle}", style="dim")

        panel = Panel(
            header_text,
            box=box.DOUBLE,
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print()

    def print_section(self, title: str, content: str = "", style: str = "blue"):
        """Imprime uma seção com título."""
        if content:
            panel = Panel(
                content,
                title=f"[bold]{title}[/bold]",
                border_style=style,
                padding=(1, 2),
            )
        else:
            panel = Panel(
                title,
                border_style=style,
                padding=(1, 2),
            )
        self.console.print(panel)

    def print_config_table(self, config: dict, title: str = "Configuração"):
        """Imprime uma tabela de configuração."""
        table = Table(
            title=title,
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            box=box.ROUNDED,
            padding=(0, 1),
        )

        table.add_column("Parâmetro", style="cyan", no_wrap=True)
        table.add_column("Valor", style="magenta")

        for key, value in config.items():
            # Formata valores booleanos e None
            if isinstance(value, bool):
                value_str = "✓" if value else "✗"
                style = "green" if value else "red"
            elif value is None:
                value_str = "N/A"
                style = "dim"
            else:
                value_str = str(value)
                style = "magenta"

            table.add_row(key, f"[{style}]{value_str}[/{style}]")

        self.console.print(table)
        self.console.print()

    def print_step(self, step: str, status: str = "info"):
        """Imprime uma etapa do processo."""
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "running": "🚀",
            "checkpoint": "📌",
            "download": "⬇️",
            "upload": "⬆️",
            "merge": "🔀",
            "convert": "⚙️",
            "save": "💾",
        }

        styles = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "running": "cyan",
            "checkpoint": "magenta",
            "download": "blue",
            "upload": "green",
            "merge": "yellow",
            "convert": "cyan",
            "save": "green",
        }

        icon = icons.get(status, "•")
        style = styles.get(status, "white")

        self.console.print(f"[{style}]{icon} {step}[/{style}]")

    def create_progress(self, description: str = ""):
        """Cria uma barra de progresso customizada."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

    def create_simple_progress(self):
        """Cria uma barra de progresso simples para downloads."""
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="bold cyan"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=False,
        )

    def print_metrics_table(self, metrics: dict, title: str = "Métricas de Treinamento"):
        """Imprime uma tabela de métricas de treinamento."""
        table = Table(
            title=title,
            show_header=True,
            header_style="bold yellow",
            border_style="yellow",
            box=box.HEAVY,
            padding=(0, 1),
        )

        table.add_column("Métrica", style="yellow", no_wrap=True)
        table.add_column("Valor", style="green", justify="right")

        for key, value in metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.6f}"
            else:
                value_str = str(value)

            table.add_row(key, value_str)

        self.console.print(table)

    def print_file_tree(self, root_path: str, files: list, title: str = "Arquivos"):
        """Imprime uma árvore de arquivos."""
        tree = Tree(
            f"[bold cyan]{title}[/bold cyan]",
            guide_style="blue",
        )

        for file in files:
            tree.add(f"[green]{file}[/green]")

        self.console.print(tree)
        self.console.print()

    def print_summary(self, summary: dict, title: str = "Resumo"):
        """Imprime um resumo final com todas as informações."""
        self.console.print()
        self.console.rule(f"[bold green]{title}[/bold green]", style="green")
        self.console.print()

        for key, value in summary.items():
            if isinstance(value, bool):
                status = "[green]✓[/green]" if value else "[red]✗[/red]"
                self.console.print(f"  {key}: {status}")
            elif isinstance(value, (int, float)):
                self.console.print(f"  {key}: [cyan]{value}[/cyan]")
            else:
                self.console.print(f"  {key}: [yellow]{value}[/yellow]")

        self.console.print()
        self.console.rule(style="green")
        self.console.print()

    def print_error(self, message: str):
        """Imprime uma mensagem de erro destacada."""
        panel = Panel(
            f"[bold red]{message}[/bold red]",
            title="[bold red]ERRO[/bold red]",
            border_style="red",
            box=box.HEAVY,
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_warning(self, message: str):
        """Imprime uma mensagem de aviso destacada."""
        panel = Panel(
            f"[bold yellow]{message}[/bold yellow]",
            title="[bold yellow]AVISO[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_success(self, message: str):
        """Imprime uma mensagem de sucesso destacada."""
        panel = Panel(
            f"[bold green]{message}[/bold green]",
            title="[bold green]SUCESSO[/bold green]",
            border_style="green",
            box=box.DOUBLE,
            padding=(1, 2),
        )
        self.console.print(panel)

    def start_timer(self):
        """Inicia o cronômetro."""
        self.start_time = time.time()

    def get_elapsed_time(self) -> str:
        """Retorna o tempo decorrido formatado."""
        if self.start_time is None:
            return "0s"

        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def print_divider(self, text: str = ""):
        """Imprime um divisor."""
        if text:
            self.console.rule(f"[bold]{text}[/bold]", style="dim")
        else:
            self.console.rule(style="dim")


class LiveMetricsDisplay:
    """Display de métricas em tempo real durante o treinamento."""

    def __init__(self):
        self.layout = Layout()
        self.metrics = {}

    def update_layout(self):
        """Atualiza o layout com as métricas atuais."""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            box=box.ROUNDED,
        )

        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="yellow", justify="right")

        for key, value in self.metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.6f}"
            else:
                value_str = str(value)
            table.add_row(key, value_str)

        self.layout.update(Panel(table, title="[bold]Métricas em Tempo Real[/bold]"))
        return self.layout

    def update_metrics(self, **kwargs):
        """Atualiza as métricas."""
        self.metrics.update(kwargs)


def format_size(bytes_size: int) -> str:
    """Formata tamanho em bytes para formato legível."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def format_time(seconds: float) -> str:
    """Formata tempo em segundos para formato legível."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
