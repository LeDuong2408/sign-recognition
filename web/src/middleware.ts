import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server"
import { NextResponse } from "next/server"
import createMiddleware from "next-intl/middleware"

import { AllLocales, AppConfig } from "./utils/AppConfig"

const intlMiddleware = createMiddleware({
  locales: AllLocales,
  localePrefix: AppConfig.localePrefix,
  defaultLocale: AppConfig.defaultLocale,
})

// Define protected routes using createRouteMatcher
const isProtectedRoute = createRouteMatcher(["/dashboard(.*)", "/onboarding(.*)"])

export default clerkMiddleware(async (auth, req) => {
  // Execute next-intl middleware first
  const intlResponse = intlMiddleware(req)

  // If intl middleware returns a response (redirect), return it
  if (intlResponse) {
    return intlResponse
  }

  // Get auth object (await the promise)
  const authObject = await auth()

  // Check if the route is protected and user is not authenticated
  if (isProtectedRoute(req) && !authObject.userId) {
    // Redirect to sign-in page
    const signInUrl = new URL("/sign-in", req.url)
    signInUrl.searchParams.set("redirect_url", req.url)
    return NextResponse.redirect(signInUrl)
  }

  // Handle organization selection logic
  const { userId, orgId } = authObject

  if (userId && !orgId && !req.nextUrl.pathname.endsWith("/onboarding/organization-selection")) {
    const organizationSelection = new URL("/onboarding/organization-selection", req.url)
    return NextResponse.redirect(organizationSelection)
  }

  // Continue with the request
  return NextResponse.next()
})

export const config = {
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always run for API routes
    "/(api|trpc)(.*)",
  ],
}
