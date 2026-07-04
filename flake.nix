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
            (python312.withPackages (ps: [
              ps.pyyaml
            ]))
            uv
            ruff
            tbb # oneTBB, runtime dep of the oidn wheel (libtbb.12.dylib)
          ];
          shellHook = ''
            # uv manages the venv and all Python deps (numpy, torch); nix pins the tools.
            export UV_PYTHON_PREFERENCE=only-system
            # The oidn wheel dlopens libtbb.12 from brew/system paths; the dyld
            # fallback lets it find nix's oneTBB instead (no brew tbb needed).
            export DYLD_FALLBACK_LIBRARY_PATH="${pkgs.tbb}/lib:''${DYLD_FALLBACK_LIBRARY_PATH:-/usr/local/lib:/usr/lib}"
          '';
        };
      });
}
