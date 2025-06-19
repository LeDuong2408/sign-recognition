import type { Config } from 'drizzle-kit';

/** @type {import('drizzle-kit').Config} */
export default {
  out: './migrations',
  schema: './src/models/Schema.ts',
  // driver: 'libsql',
  dialect: "sqlite",
  driver: "d1-http",
  // dbCredentials: {
  //   // url: process.env.DATABASE_URL ?? '',
  //   // wranglerConfigPath: "./wrangler.toml",
  //   // dbName: "coolkatznotrealdbname",
  // },
} satisfies Config;
