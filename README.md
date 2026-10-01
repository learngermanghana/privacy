# Falowen — Legal / Enrollment Agreement & Policies (GitHub Pages)

This package serves a single public reference page at **legal.falowen.app** for the Falowen Enrollment Agreement, Payment Agreement, Privacy Policy, Terms of Service, enrollment guidance, FAQ, and contact information. Student registration and class-specific enrollment data live in Falowen; this GitHub Pages site does not maintain a separate student registration system.

## Deploy
1. Create/choose your GitHub repo and upload these files.
2. In **Settings → Pages**, select your branch/folder. Under **Custom domain**, enter:
```
legal.falowen.app
```
3. Ensure a `CNAME` file exists in the repo root with exactly that hostname (GitHub can create it for you when you save the custom domain).

## Cloudflare DNS (authoritative)
Add a CNAME:
- **Type:** CNAME
- **Name:** register
- **Target:** learngermanghana.github.io
- **Proxy:** DNS only (gray cloud)
- **TTL:** Auto

## Optional redirect from old subdomain
In Cloudflare → **Rules → Redirect Rules → Create**:
- If Hostname equals `legal.falowen.app`
- Static redirect to `https://legal.falowen.app` with status 301

## Social image
Replace `og-image.png` with a 1200×630 image for rich previews (OG/Twitter).
