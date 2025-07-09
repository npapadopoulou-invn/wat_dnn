from rich import print
from rich.console import Console

def ascii_border(text, style="classic", color="cyan"):
    console = Console()
    length = len(text) + 4
    if style == "classic":
        print(f"[{color}]+{'-' * length}+[/]")
        print(f"[{color}]|  [bold]{text}[/bold]  |[/]")
        print(f"[{color}]+{'-' * length}+[/]")
    elif style == "double":
        print(f"[{color}]╔{'═' * length}╗[/]")
        print(f"[{color}]║  [bold]{text}[/bold]  ║[/]")
        print(f"[{color}]╚{'═' * length}╝[/]")


