{
  description = "nrp: sample reimplementation of Neural Render Proxies (Sancho et al., EGSR 2026)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python312
            uv
            ruff
          ];
          shellHook = ''
            # uv manages the venv and all Python deps (numpy, torch); nix pins the tools.
            export UV_PYTHON_PREFERENCE=only-system
          '';
        };
      });
}
