"""`python -m runkit ...` -- the same as the `runkit` command. The relaunch under
uv uses this form, so it does not depend on the console script being installed
in the experiment's environment."""
from .cli import app

app()
