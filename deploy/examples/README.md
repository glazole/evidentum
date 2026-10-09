# Deployment examples

The full Compose setup preserves the original HTTPS and Certbot renewal arrangement.
Set `DOMAIN` in `.env`; the nginx image substitutes it into
`nginx/templates/default.conf.template` at startup. `NGINX_ENVSUBST_FILTER=DOMAIN`
keeps nginx variables such as `$host` and `$request_uri` intact. A single server
configuration is generated, avoiding the previous duplicate configuration files.

`nginx-https.conf.example` is the same configuration with a sample `example.org`
domain, for reference. It is not the active configuration.

Supply certificates at `/etc/letsencrypt/live/<DOMAIN>/` in the `letsencrypt` volume
before starting nginx. The Certbot service renews existing certificates but does not
bootstrap the first certificate; configure first issuance separately. Reload nginx
after certificate renewal. This keeps the original deployment's TLS prerequisite.

The ordinary `docker-compose.yml` does not require a domain, nginx or certificates.
It exposes API and UI ports on localhost and uses its own internal bridge network.

The two Compose variants reuse container and volume names. Choose one at a time.
Redis remains in the original full deployment, although application code currently
does not consume it. Administrative tokens do not protect every query/read endpoint
or implement per-user isolation.
