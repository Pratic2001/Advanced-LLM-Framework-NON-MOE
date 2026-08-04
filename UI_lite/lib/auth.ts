import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { CredentialsSignin } from "next-auth";
import { compare } from "bcryptjs";
import prisma from "./db";

// In NextAuth v5 (Auth.js), `trustHost` must be enabled when running on
// localhost or any host that is not a known Vercel/Netlify/etc. platform.
// Without it, every /api/auth/* call throws `UntrustedHost: Host must be
// trusted`. We default to true and allow `AUTH_TRUST_HOST=false` to opt out
// for production deployments behind a known proxy.
export const { handlers, auth, signIn, signOut } = NextAuth({
  trustHost: (process.env.AUTH_TRUST_HOST ?? "true") !== "false",
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          throw new CredentialsSignin("Email and password required");
        }

        const user = await prisma.user.findUnique({
          where: { email: credentials.email as string },
        });

        if (!user) {
          throw new CredentialsSignin("Invalid email or password");
        }

        const isValid = await compare(
          credentials.password as string,
          user.passwordHash
        );

        if (!isValid) {
          throw new CredentialsSignin("Invalid email or password");
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
        };
      },
    }),
  ],
  session: {
    strategy: "jwt",
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },
  pages: {
    signIn: "/login",
    newUser: "/signup",
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string;
      }
      return session;
    },
  },
});
