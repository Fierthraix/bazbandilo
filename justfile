default: make

alias b := ber
alias c := cfar
alias f := foo
alias g := goo
alias i := interactive
alias j := jupyter
alias m := make
alias p := plop
alias pv := plop_view
alias s := shell
alias t := test

test_dir := justfile_directory() + "/tests/"
ffi_dir := justfile_directory() + "/bazbandilo"
python_dir := justfile_directory() + "/pitono/"

ber:
	cargo test --test ber -- --nocapture

[no-cd]
cfar *args:
	poetry run python3 -i {{ python_dir + "cfar.py" }} {{ args }}

clean:
	-rm "{{ ffi_dir }}"/*.so

make:
	poetry run maturin develop

flamegraph arg:
	poetry run python3 -m plop.collector -f flamegraph {{ arg }}

[no-cd]
foo *args:
	poetry run python3 -i {{ python_dir + "foo.py" }} {{ args }}

[no-cd]
goo *args:
	poetry run python3 -i {{ python_dir + "goo.py" }} {{ args }}

install:
	poetry install
	cargo install maturin
	@just make

jupyter:
	poetry run jupyter lab

plop arg:
	poetry run python3 -m plop.collector {{ arg }}

plop_view:
	poetry run python3 -m plop.viewer --datadir="{{ justfile_directory() + '/kom_py/profiles' }}"

[no-cd]
shell *kargs:
	poetry run python3 {{ kargs }}

[no-cd]
interactive *kargs:
	poetry run python3 -i {{ kargs }}

test arg:
	cargo test --test {{ arg }} -- --nocapture
