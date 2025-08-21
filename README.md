# Falowen — Register subdomain (GitHub Pages)

This package serves a single page at **register.falowen.app** with: Register, Payment Agreement, Privacy, and Terms.

## Deploy
1. Create/choose your GitHub repo and upload these files.
2. In **Settings → Pages**, select your branch/folder. Under **Custom domain**, enter:
```
register.falowen.app
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
- Static redirect to `https://register.falowen.app` with status 301

## Social image
Replace `og-image.png` with a 1200×630 image for rich previews (OG/Twitter).
