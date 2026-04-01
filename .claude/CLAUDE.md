# Scholarflow - Claude Code & GLM-4.7 Guidelines

## 🎯 Project Overview

**AI-Powered Learning Platform** with three main components:

- `scholarflow-be/` - NestJS Backend (TypeScript)
- `scholarflow-fe/` - Next.js Frontend (TypeScript)
- `ace-main-1/` - Python AI/ML (ACE System)

---

## 🛡️ CRITICAL Rules for GLM-4.7 Stability

### File Editing Rules (MUST FOLLOW)

1. **ALWAYS Read before Edit** - Never edit a file without reading it first
2. **Use Relative Paths** - Never use absolute paths like `/Users/macbook/...`
3. **Make Small, Targeted Edits** - One logical change per edit operation
4. **Verify After Editing** - Read the file again to confirm changes
5. **If Edit Tool Fails** - Fall back to Bash with sed/python script:
   ```bash
   # Example fallback for file editing
   py -c "
   with open('path/to/file.ts', 'r') as f:
       content = f.read()
   content = content.replace('old_text', 'new_text')
   with open('path/to/file.ts', 'w') as f:
       f.write(content)
   "
   ```

### Path Examples

```bash
# ✅ CORRECT - Relative paths
scholarflow-be/src/app.service.ts
scholarflow-fe/src/app/page.tsx
ace-main-1/ace/main.py

# ❌ WRONG - Absolute paths
/Users/macbook/Desktop/scholarflow/scholarflow-be/src/app.service.ts
```

---

## 🏗️ Architecture Guidelines

### NestJS Backend (`scholarflow-be/`)

- **Module-based architecture**: Each feature = 1 module
- **Use DTOs with class-validator** for all endpoints
- **Dependency injection pattern** for services
- **BullMQ** for background jobs
- **TypeORM** with PostgreSQL
- **Jest** for testing

### Next.js Frontend (`scholarflow-fe/`)

- **App Router** structure (not Pages Router)
- **Server Components** by default, Client Components when needed
- **Zustand** for state management
- **Tailwind CSS** for styling
- **Monaco Editor** for IDE functionality
- **next-intl** for i18n (en, ko)

### Python AI/ML (`ace-main-1/`)

- **Use type hints** everywhere
- **Follow PEP 8** style guide
- **Use virtual environments** (venv/conda)
- **Document functions** with docstrings
- **OpenAI/Together/SambaNova** APIs for AI operations
- **FAISS** for vector operations

---

## 📁 Working Directory Structure

```
scholarflow/
├── scholarflow-be/          # NestJS Backend
│   ├── src/
│   │   ├── api/            # API routes & controllers
│   │   ├── database/       # TypeORM entities, migrations
│   │   ├── background/     # BullMQ jobs
│   │   ├── mail/          # Email (Mailgun)
│   │   ├── redis/         # Redis config
│   │   ├── config/        # Configuration
│   │   ├── decorators/    # Custom decorators
│   │   ├── guards/        # Auth guards (JWT)
│   │   └── utils/         # Utility functions
│   └── test/              # Jest tests
├── scholarflow-fe/          # Next.js Frontend
│   ├── src/
│   │   ├── app/           # App Router pages
│   │   │   ├── ide/       # IDE interface
│   │   │   ├── courses/   # Course pages
│   │   │   ├── admin/     # Admin dashboard
│   │   │   ├── dashboard/ # User dashboard
│   │   │   └── ai-builder/# AI course builder
│   │   ├── types/         # TypeScript types
│   │   └── messages/      # i18n (en, ko)
│   └── public/            # Static assets
└── ace-main-1/             # Python AI/ML
    ├── ace/               # Main ACE module
    └── eval/              # Evaluation scripts
```

---

## 🔧 Development Workflow

### 1. Before Making Changes

- Read the file(s) you need to modify
- Understand the existing patterns
- Check for related files (imports, dependencies)

### 2. Making Changes

- Start with backend changes (NestJS), then frontend (Next.js)
- Update DTOs when modifying API endpoints
- Keep changes small and focused
- Test after each significant change

### 3. Testing

```bash
# Backend
cd scholarflow-be && npm test

# Frontend
cd scholarflow-fe && npm test

# Python
cd ace-main-1 && pytest
```

### 4. Linting

```bash
# Backend
cd scholarflow-be && npm run lint

# Frontend
cd scholarflow-fe && npm run lint

# Python
cd ace-main-1 && ruff check .
```

---

## 🚨 Common Pitfalls to Avoid

### Don't:

- Edit multiple files in parallel without reading them first
- Use absolute paths
- Make large, sweeping edits in one operation
- Skip type definitions in TypeScript
- Forget to update imports when moving files
- Mix server and client components in Next.js without 'use client'

### Do:

- Read files before editing
- Use relative paths from project root
- Make incremental changes
- Follow existing patterns in the codebase
- Test after changes
- Use proper TypeScript types
- Keep AI/ML Python code well-documented

---

## 📋 Quick Reference

### Running the Project

```bash
# Backend
cd scholarflow-be && npm run start:dev

# Frontend
cd scholarflow-fe && npm run dev

# Python AI/ML
cd ace-main-1 && python -m ace.main
```

### Key Technologies

| Component | Tech       | Purpose                     |
| --------- | ---------- | --------------------------- |
| Backend   | NestJS     | REST API, WebSockets        |
| Frontend  | Next.js 16 | SSR, App Router             |
| AI/ML     | Python     | ACE System, LLM integration |
| Database  | PostgreSQL | Primary data store          |
| Cache     | Redis      | Sessions, BullMQ            |
| Auth      | JWT        | Authentication              |
| Payment   | PayPal     | Payment processing          |

---

## 🔗 External Services

- **Mailgun** - Email delivery
- **OpenAI API** - GPT models
- **Together AI** - Alternative LLMs
- **SambaNova AI** - AI inference
- **Google OAuth** - Authentication
- **PayPal** - Payments

---

## 🤖 Installed MCP Servers

### Core MCP Servers

| MCP Server     | Purpose                                      | Command                                      |
| -------------- | -------------------------------------------- | -------------------------------------------- |
| **Playwright** | Browser automation, testing, scraping        | `npx @playwright/mcp@latest`                 |
| **GitHub**     | PR, issues, code review, repo management     | `npx -y @modelcontextprotocol/server-github` |
| **Firecrawl**  | Web scraping, crawling, research             | `npx -y firecrawl-mcp`                       |
| **DBHub**      | PostgreSQL, MySQL, SQLite management         | `npx -y @bytebase/dbhub-mcp`                 |
| **Supabase**   | Full Supabase management (DB, Auth, Storage) | `npx -y @supabase/mcp-server-supabase`       |
| **Context7**   | Read docs realtime (NestJS, Python libs)     | `npx -y @upstash/context7-mcp@latest`        |
| **Composio**   | 1000+ integrations (Slack, Notion, Gmail...) | `npx -y @composio/mcp`                       |
| **FastMCP**    | Build custom Python MCP servers              | `pip install fastmcp`                        |

### MCP Usage Notes

- MCP servers use lazy loading via `ENABLE_TOOL_SEARCH=true`
- Auto-compact enabled at 180K tokens / 80% context
- Post-edit hooks auto-format with Prettier

---

## 🛠️ Installed Claude Code Skills

### Anthropic Official Skills

| Skill              | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| **pdf**            | Read, extract, create, merge/split PDF, fill forms  |
| **xlsx**           | Spreadsheet: formulas, charts, data transformations |
| **docx**           | Word docs: create, edit, tracked changes, comments  |
| **pptx**           | PowerPoint: create/edit presentations               |
| **webapp-testing** | Test web apps with Playwright                       |
| **mcp-builder**    | Build MCP servers to expose APIs                    |

### Community Skills

| Skill                        | Purpose                                                                 |
| ---------------------------- | ----------------------------------------------------------------------- |
| **Superpowers** (obra)       | Framework agentic: brainstorm → spec → plan → subagent → review → merge |
| **claude-scientific-skills** | 125+ scientific skills: bioinformatics, ML, specialized databases       |

---

## ⚡ Token Optimization Settings

### Auto-Compact Configuration

- `CLAUDE_CODE_AUTO_COMPACT_WINDOW`: 180000 tokens
- `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`: 80%
- `ENABLE_TOOL_SEARCH`: true (lazy MCP loading)

### Hooks

- **PostToolUse**: Auto-format with Prettier on Edit/Write

### Model Configuration

- Haiku: `glm-4.5-air` (cost-optimized)
- Sonnet: `glm-4.7` (balanced)
- Opus: `glm-5.1` (quality)
- Base URL: `https://api.z.ai/api/anthropic`

---

_Generated for Scholarflow AI-Powered Learning Platform_
_Optimized for Claude Code with GLM-4.7_
_Updated: 2025-04-01_
