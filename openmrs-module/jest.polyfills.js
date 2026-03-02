/**
 * jest.polyfills.js
 *
 * Runs via jest.config.js `setupFiles` — before jsdom initialises each test file.
 * Polyfills Node.js globals that jsdom does not expose in the test environment.
 */
const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// Web Streams API — needed for SSE (ReadableStream.getReader()) tests.
// Available in Node ≥ 18 via stream/web; jsdom does not re-export them.
const { ReadableStream, WritableStream, TransformStream } = require('stream/web');
if (!global.ReadableStream) global.ReadableStream = ReadableStream;
if (!global.WritableStream) global.WritableStream = WritableStream;
if (!global.TransformStream) global.TransformStream = TransformStream;
