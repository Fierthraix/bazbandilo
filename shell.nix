{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    just
    poetry
    maturin
    rustc
    cargo
    python312Full
    python312Packages.scipy
    python312Packages.matplotlib
  ];
}
