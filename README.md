# Dependencies:

* [Rust](https://rustup.rs/) 
* [python](https://www.python.org/) 
  * [`matplotlib`](https://pypi.org/project/matplotlib/)
  * [`numpy`](https://pypi.org/project/numpy/)

# Documentation:

[docs.rs/bazbandilo](https://docs.rs/bazbandilo/latest/bazbandilo/)

# To create the graphics.

```bash
cargo test  # And check the `/tmp/` directory
            # in your favourite file explorer.
```

# To create **all** the graphics.
```bash
cargo test -- --include-ignored
```
