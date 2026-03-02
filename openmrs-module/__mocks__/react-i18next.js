/**
 * Manual mock for react-i18next.
 *
 * Referenced via jest.config.js moduleNameMapper so it applies to every test
 * without needing an explicit jest.mock() call in each file.
 *
 * t(key, fallback) returns the fallback string when provided, otherwise the key.
 * This lets component tests use the same plain-English strings they render.
 */
module.exports = {
  useTranslation: () => ({
    t: (key, fallback, _opts) => {
      if (typeof fallback === 'string') return fallback;
      return key;
    },
    i18n: {
      changeLanguage: () => Promise.resolve(),
      language: 'en',
    },
  }),
  Trans: ({ children }) => children,
  initReactI18next: { type: '3rdParty', init: () => {} },
};
