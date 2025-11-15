# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of PrivaChat Agents seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities via:

1. **GitHub Security Advisories** (Preferred)
   - Navigate to the Security tab in the repository
   - Click "Report a vulnerability"
   - Fill out the form with details

2. **Email** (Alternative)
   - Send an email to: security@privachat.example.com
   - Include "SECURITY" in the subject line
   - Provide detailed information about the vulnerability

### What to Include

When reporting a vulnerability, please include:

- **Description**: Clear description of the vulnerability
- **Impact**: What can an attacker do? What is affected?
- **Reproduction steps**: Step-by-step instructions to reproduce
- **Affected versions**: Which versions are impacted?
- **Suggested fix**: If you have ideas for mitigation
- **Your contact info**: For follow-up questions

Example report:
```
Title: SQL Injection in search endpoint

Description:
The /api/v1/search endpoint is vulnerable to SQL injection through
the 'query' parameter when using raw SQL queries.

Impact:
An attacker could extract sensitive data from the database or
execute arbitrary SQL commands.

Steps to Reproduce:
1. Send POST request to /api/v1/search
2. Include payload: {"query": "test' OR '1'='1"}
3. Observe database error revealing schema

Affected Versions:
0.1.0 - 0.1.3

Suggested Fix:
Use parameterized queries or ORM (SQLAlchemy) for all database access.
```

### Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix Development**: Depends on severity (1-4 weeks)
- **Disclosure**: After fix is released and deployed

### Severity Levels

We use the following severity classifications:

**Critical** (Fix within 1-3 days)
- Remote code execution
- Authentication bypass
- Privilege escalation
- Data breach

**High** (Fix within 1 week)
- SQL injection
- Cross-site scripting (XSS)
- Insecure authentication
- Information disclosure

**Medium** (Fix within 2-4 weeks)
- CSRF vulnerabilities
- Insecure direct object references
- Missing security headers
- Weak cryptography

**Low** (Fix in next release)
- Rate limiting issues
- Information leakage
- Security misconfigurations

### Disclosure Policy

- **Coordinated disclosure**: We work with reporters to understand and fix issues before public disclosure
- **Credit**: Security researchers who responsibly disclose vulnerabilities will be credited (if desired)
- **Public disclosure**: After fix is released and users have time to update (typically 2-4 weeks)
- **CVE assignment**: We will request CVE IDs for significant vulnerabilities

### Security Best Practices

When using PrivaChat Agents:

**Environment Variables:**
- Never commit `.env` files
- Use strong, unique API keys
- Rotate keys regularly
- Use different keys for dev/prod

**Docker Security:**
- Run containers as non-root user
- Keep base images updated
- Scan images for vulnerabilities
- Use Docker secrets for sensitive data

**API Security:**
- Enable authentication in production
- Use HTTPS/TLS for all connections
- Implement rate limiting
- Validate all inputs
- Sanitize outputs

**Database Security:**
- Use strong database passwords
- Enable SSL/TLS for connections
- Regular backups
- Restrict network access
- Apply security updates promptly

**Network Security:**
- Use private networks for services
- Implement firewall rules
- Restrict public endpoints
- Use VPN for remote access

### Known Security Considerations

**Third-Party Dependencies:**
- We use multiple external services (OpenRouter, SearxNG, Langfuse)
- Review their security policies before use
- API keys grant access to these services
- Monitor usage for anomalies

**Vector Database:**
- pgvector stores embeddings and documents
- Sensitive data may be stored in vectors
- Use encryption at rest for PostgreSQL
- Implement access controls

**LLM Security:**
- Language models can be influenced by input
- Implement prompt injection protections
- Sanitize user inputs
- Monitor for abuse patterns

**Web Scraping:**
- Crawl4AI may access sensitive sites
- Respect robots.txt
- Implement authentication if needed
- Be aware of legal implications

### Security Updates

Subscribe to security advisories:
- Watch the GitHub repository
- Enable security alerts
- Check release notes for security fixes
- Subscribe to mailing list (if available)

### Bug Bounty Program

We currently do not have a formal bug bounty program. However:
- We appreciate responsible disclosure
- We will credit researchers in release notes
- We may offer recognition or swag
- Contact us for collaboration opportunities

### Security Audits

Last security audit: Not yet conducted (project is new)

Planned audits:
- Internal code review: Before v1.0 release
- Third-party audit: After v1.0 with sufficient funding
- Dependency scanning: Automated via GitHub Dependabot

### Compliance

**Data Protection:**
- No PII is collected by default
- Search queries may be logged (configurable)
- LLM providers may retain data (check their policies)
- GDPR considerations apply if deployed in EU

**API Key Management:**
- Keys are stored in environment variables
- Not logged or exposed in responses
- Transmitted over HTTPS only
- Rotatable without code changes

### Contact

For security-related questions:
- Security issues: Use GitHub Security Advisories
- General security questions: Open GitHub issue with `security` label
- Private inquiries: security@privachat.example.com

### Hall of Fame

Security researchers who have responsibly disclosed vulnerabilities:

_(None yet - be the first!)_

---

Thank you for helping keep PrivaChat Agents and our community safe! 🔒
