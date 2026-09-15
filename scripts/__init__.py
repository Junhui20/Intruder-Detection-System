"""One-time setup helpers. Everything else the app does for itself on first run
(directories, database schema, optimized model exports).

- setup_secure_config.py: writes .env (Telegram token, web UI password)
- setup_ollama.py: installs Ollama and pulls the caption model for a tier
"""
