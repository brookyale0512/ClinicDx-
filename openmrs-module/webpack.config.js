const openmrsConfig = require('openmrs/default-webpack-config');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');

module.exports = (env, argv) => {
  const config = openmrsConfig(env, argv);

  // Replace the ForkTsCheckerWebpackPlugin to also exclude errors from
  // @openmrs node_modules source files (which use implicit-any patterns
  // incompatible with our strict tsconfig).
  const idx = config.plugins.findIndex((p) => p instanceof ForkTsCheckerWebpackPlugin);
  if (idx !== -1) {
    config.plugins[idx] = new ForkTsCheckerWebpackPlugin({
      issue: {
        exclude: [
          { severity: 'error', code: 'TS2786' },
          { file: '**/node_modules/**' },
        ],
      },
    });
  }

  return config;
};
