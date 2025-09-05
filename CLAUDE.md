# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a cryptocurrency trading bot system focused on multi-exchange trading with advanced risk protection. The codebase consists of two main components:

- **stargate_all_in_one.py**: Main trading server (v4.2) with enhanced risk protection system
- **gpt_patcher.py**: AI-powered code modification tool using OpenAI GPT models

## Architecture

### Core Trading System (stargate_all_in_one.py)
- Multi-exchange support (Bybit, Bitget) with unified adapter interfaces
- Flask web server for webhook endpoints and GUI interface
- Risk protection system with retry-based protective exits
- Trailing stop-loss functionality with configuration-driven reverse logic
- Multi-process architecture with GUI (when available) and headless modes

### Key Components:
- Exchange adapters with normalized symbol handling
- Price utilities and VWAP calculations  
- Configuration management with trailing/reverse logic
- Protective exit enforcement system
- Web interface for settings and status

### GPT Patcher System (gpt_patcher.py)
- Automated code modification using OpenAI API
- Support for GPT-4, GPT-5, and other models
- Token budget management and context optimization
- Direct plan execution from JSON instructions
- File tracking and modification logging

## Development Commands

### Running the Trading Server
```bash
# Run with GUI (if available)
python stargate_all_in_one.py

# Run in headless mode
python stargate_all_in_one.py --headless

# Run with specific configuration
python stargate_all_in_one.py --config config.json
```

### GPT Patcher Usage
```bash
# Set required environment variables
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-5-mini"  # or other supported models

# Run patcher with instructions
export USER_INSTRUCTIONS="modification instructions or JSON plan"
python gpt_patcher.py
```

### Dependencies
Install Python dependencies:
```bash
pip install -r stargate-trading-bot/requirements.txt
```

Core dependencies: flask, requests, anthropic

## GitHub Workflows

### Automated Code Modification (.github/workflows/auto-commit.yml)
- Triggers on push to main branch or workflow dispatch
- Uses GPT models for automated code changes
- Creates PRs with modifications
- Supports auto-merge with [AUTO_MERGE] tag

### Trailing Reverse Patch (.github/workflows/apply-trailing-reverse-patch.yml)  
- Applies specific patches for trailing stop functionality
- Injects helper functions and reverse logic
- Creates PRs for trailing configuration changes

## Configuration

### Trading Bot Configuration
The trading system uses configuration dictionaries with keys:
- `enable_trailing` / `trailing_enabled`: Enable/disable trailing stops
- `trailing.enabled`: Nested trailing configuration
- Exchange-specific settings for Bybit and Bitget

### GPT Patcher Configuration
Environment variables:
- `OPENAI_MODEL`: Model selection (gpt-5-mini, gpt-4o, etc.)
- `MAX_FILES`: File processing limit
- `TOKEN_BUDGET`: Context token limit
- `USER_INSTRUCTIONS`: Modification instructions or JSON plan

## Key Functions and Locations

### Trading Logic
- `round_to_tick()`: Price rounding utilities (stargate_all_in_one.py:64)
- `trailing_enabled_from_cfg()`: Configuration parsing for trailing stops
- Exchange adapter classes with unified interfaces

### Risk Management
- `enforce_protective_exits()`: Retry-based protection system
- `reverse_on_opposite`: Logic for reverse trading on trailing disable

### Web Endpoints
- `/webhook`: Main trading signal processing
- `/settings`: Configuration management
- `/settings/trailing`: Trailing-specific settings

## Important Notes

- The system includes extensive Korean comments and documentation
- Risk protection is a core focus with multiple safety layers
- The codebase supports both GUI and headless deployment modes
- GitHub Actions provide automated code modification capabilities
- Configuration changes can trigger reverse trading logic