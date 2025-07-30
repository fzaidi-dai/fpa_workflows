# Security Guidelines

This document outlines security best practices for the FPA Agents project to prevent accidental exposure of sensitive information.

## Environment Variables Security

### ⚠️ CRITICAL: Never Commit .env Files

**The .env file contains sensitive API keys and credentials that must NEVER be committed to version control.**

### What Happened (Incident Report)
- Date: July 30, 2025
- Issue: .env file was accidentally committed to GitHub repository
- Exposed: Google API Key and OpenRouter API Key
- Resolution: Keys revoked, Git history cleaned, security measures implemented

### Prevention Measures Implemented

#### 1. .gitignore Protection
The following patterns are now in .gitignore to prevent environment file commits:
```
# Environment files - NEVER COMMIT THESE!
.env
.env.local
.env.*.local
.env.development
.env.production
.env.staging
.env.test
```

#### 2. Template System
- **Use**: `.env.template` - Safe template with placeholder values
- **Never commit**: `.env` - Contains actual sensitive values

#### 3. Setup Process
```bash
# For new developers
cp .env.template .env
# Edit .env with your actual API keys
# NEVER commit .env to Git
```

## API Key Management

### Immediate Actions for Exposed Keys
1. **Revoke immediately** - Disable the exposed keys in their respective services
2. **Generate new keys** - Create fresh API keys
3. **Update .env** - Replace with new keys locally
4. **Monitor usage** - Check for unauthorized access

### Key Rotation Schedule
- **Google API Keys**: Rotate every 90 days
- **OpenRouter API Keys**: Rotate every 90 days
- **Other API Keys**: Follow service-specific recommendations

### Secure Storage
- **Local Development**: Use .env files (never committed)
- **Production**: Use environment variables or secure secret management
- **CI/CD**: Use encrypted secrets in GitHub Actions/CI systems

## Git Security Best Practices

### Pre-commit Checks
Consider installing pre-commit hooks to prevent accidental commits:
```bash
# Install pre-commit
pip install pre-commit

# Add to .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: detect-private-key
```

### Repository Scanning
- Use tools like `git-secrets` or `truffleHog` to scan for secrets
- Enable GitHub's secret scanning alerts
- Regular security audits of commit history

## Emergency Response Plan

### If Secrets Are Exposed
1. **Immediate Response** (within 5 minutes)
   - Revoke all exposed API keys
   - Change any exposed passwords
   - Notify team members

2. **Git Cleanup** (within 30 minutes)
   - Remove secrets from Git history using `git filter-branch`
   - Force push to overwrite remote history
   - Verify complete removal

3. **Monitoring** (ongoing)
   - Monitor API usage for unauthorized access
   - Check service logs for suspicious activity
   - Update incident documentation

### Contact Information
- **Security Lead**: [Add contact information]
- **Emergency Response**: [Add emergency contact]

## Development Guidelines

### Environment Setup
```bash
# 1. Clone repository
git clone <repository-url>

# 2. Copy environment template
cp .env.template .env

# 3. Edit with your keys (NEVER commit this file)
nano .env

# 4. Verify .env is in .gitignore
git check-ignore .env  # Should return .env
```

### Code Review Checklist
- [ ] No hardcoded API keys or secrets
- [ ] .env files not included in commits
- [ ] Sensitive data properly externalized
- [ ] Environment variables documented in .env.template

### Testing with Secrets
- Use test/sandbox API keys for development
- Never use production keys in development
- Mock external services when possible

## Compliance and Auditing

### Regular Security Reviews
- Monthly review of .gitignore effectiveness
- Quarterly API key rotation
- Annual security audit of entire codebase

### Documentation Requirements
- All new environment variables must be documented in .env.template
- Security incidents must be logged in this document
- Changes to security practices require team approval

## Tools and Resources

### Recommended Tools
- **git-secrets**: Prevents committing secrets
- **pre-commit**: Automated pre-commit checks
- **1Password CLI**: Secure secret management
- **GitHub Secret Scanning**: Automatic detection

### External Resources
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [OWASP Secrets Management](https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_password)
- [Git Security Best Practices](https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage)

## Incident History

### 2025-07-30: .env File Exposure
- **Severity**: High
- **Affected**: Google API Key, OpenRouter API Key
- **Resolution**: Keys revoked, Git history cleaned, security measures implemented
- **Lessons Learned**: Need for better pre-commit hooks and developer training

---

**Remember: Security is everyone's responsibility. When in doubt, ask the security team.**
