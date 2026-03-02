/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'jsdom',
  transform: {
    '^.+\\.(t|j)sx?$': ['@swc/jest', {
      jsc: {
        parser: { syntax: 'typescript', tsx: true },
        transform: { react: { runtime: 'classic' } },
      },
    }],
  },
  // setupFiles runs before jsdom — use for polyfills (TextEncoder, etc.)
  setupFiles: ['<rootDir>/jest.polyfills.js'],
  // setupFilesAfterEnv runs after jsdom — use for jest-dom matchers
  setupFilesAfterEnv: ['@testing-library/jest-dom'],
  moduleNameMapper: {
    '\\.scss$': 'identity-obj-proxy',
    // Route react-i18next to our shim so t(key, fallback) returns the fallback
    '^react-i18next$': '<rootDir>/__mocks__/react-i18next.js',
  },
  testMatch: ['**/*.test.(ts|tsx)'],
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/declarations.d.ts',
  ],
};
