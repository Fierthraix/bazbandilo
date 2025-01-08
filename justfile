default: make

alias b := ber
alias c := clean
alias f := foo
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

clean:
	-rm "{{ ffi_dir }}"/*.so

make:
	poetry run maturin develop

flamegraph arg:
	poetry run python3 -m plop.collector -f flamegraph {{ arg }}

[no-cd]
foo *args:
	poetry run python3 -i {{ python_dir + "foo.py" }} {{ args }}

install:
	poetry install
	cargo install maturin
	@just make

jupyter:
	poetry run jupyter lab

pd:
	cargo test --test pd -- --nocapture

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

run:
	# @just ber
	# cp /tmp/bers.json bers_curr.json
	# cp /tmp/bers.msgpack bers_curr.msgpack
	@just pd
	cp /tmp/results.json results_curr.json
	cp /tmp/results.msgpack results_curr.msgpack
	@just i ./pitono/foo.py
