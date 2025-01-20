# Bluesky Bot

A Node.js-based bot for Bluesky social network that uses AI services (Claude, Gemini) for interaction.

## Features

- Automated interaction with Bluesky posts
- Integration with multiple AI services:
  - Anthropic's Claude
  - Google's Gemini
  - FAL.ai for image processing
- Configurable response intervals and retry mechanisms
- Production-ready with Render deployment support

## Prerequisites

- Node.js >= 18.0.0
- A Bluesky account
- API keys for:
  - Anthropic Claude
  - Google Gemini
  - FAL.ai

## Installation

1. Clone the repository:

bash
git clone [your-repo-url]
cd [your-repo-name]

2. Install dependencies:

bash
npm install

3. Create a `.env` file in the root directory with the following variables:

env
BLUESKY_IDENTIFIER=your.handle.bsky.social
BLUESKY_APP_PASSWORD=your-app-password
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key
FAL_API_KEY=your-fal-key
Optional configurations
CHECK_INTERVAL=60000
MAX_RETRIES=5
BACKOFF_DELAY=60000
MAX_REPLIED_POSTS=1000

## Usage

To run locally:

bash
npm start


## Deployment

This project is configured for deployment on Render. To deploy:

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect to your GitHub repository
4. Set the required environment variables in the Render dashboard
5. Deploy!

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| BLUESKY_IDENTIFIER | Your Bluesky handle | Yes |
| BLUESKY_APP_PASSWORD | Your Bluesky app password | Yes |
| ANTHROPIC_API_KEY | API key for Claude | Yes |
| GEMINI_API_KEY | API key for Gemini | Yes |
| FAL_API_KEY | API key for FAL.ai | Yes |
| CHECK_INTERVAL | Interval between checks (ms) | No |
| MAX_RETRIES | Maximum retry attempts | No |
| BACKOFF_DELAY | Delay between retries (ms) | No |
| MAX_REPLIED_POSTS | Maximum posts to track | No |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

ISC

## Security

⚠️ Never commit your `.env` file or expose your API keys. Make sure to add `.env` to your `.gitignore` file.#   d e a r e s t - c l a u d e  
 