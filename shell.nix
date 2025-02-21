{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    just
    uv
    maturin
    rustc
    cargo
    python312Full
    python312Packages.scipy
    python312Packages.matplotlib
  ];
}
