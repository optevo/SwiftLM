{
  description = "SwiftLM development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          cmake  # required by build.sh to compile mlx.metallib Metal kernels
        ];

        # Swift and Metal toolchain come from Xcode — not managed here.
        # DEVELOPER_DIR/SDKROOT are restored in .envrc after `use flake`.
      };
    };
}
