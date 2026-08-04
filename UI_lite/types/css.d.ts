/**
 * Allow static CSS imports for libraries like @xterm/xterm that ship a
 * stylesheet. Webpack/Turbopack handles these at build time.
 */
declare module "*.css";