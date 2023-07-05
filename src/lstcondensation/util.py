import coolname
from rich import Console


def random_trial_name() -> str:
    c = Console(width=80)
    name = coolname.generate_slug(3)
    c.rule(name)
    return name
