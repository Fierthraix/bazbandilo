default: make

alias b := ber
alias c := cfar
alias i := interactive
alias j := jupyter
alias m := make
alias s := shell
alias t := test

test_dir := justfile_directory() + "/tests/"
ffi_dir := justfile_directory() + "/bazbandilo"
python_dir := justfile_directory() + "/pitono/"

ber:
	cargo test --test ber -- --nocapture

[no-cd]
cfar *args:
	uv run python3 -i {{ python_dir + "cfar.py" }} {{ args }}

clean:
	-rm "{{ ffi_dir }}"/*.so

make:
	uv run maturin develop

[no-cd]
interactive *kargs:
	uv run python3 -i {{ kargs }}

jupyter:
	uv run jupyter lab

[no-cd]
shell *kargs:
	uv run python3 {{ kargs }}

test arg:
	cargo test --test {{ arg }} -- --nocapture
