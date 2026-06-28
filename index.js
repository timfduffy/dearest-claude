import { AtpAgent } from '@atproto/api';
import { Anthropic } from '@anthropic-ai/sdk';
import config from './config.js';
import fetch from 'node-fetch';
import { fal } from "@fal-ai/client";
import sharp from 'sharp';
import express from 'express';
import { OpenAI } from 'openai';

// Add express to your package.json dependencies
const app = express();
const PORT = process.env.PORT || 3000;

// Add basic health check endpoint
app.get('/', (req, res) => {
  res.send('Bot is running!');
});

app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

// Start express server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

// Initialize Bluesky client
const agent = new AtpAgent({
  service: 'https://bsky.social',
});

// Initialize Anthropic client
const anthropic = new Anthropic({
  apiKey: config.ANTHROPIC_API_KEY,
});

// Initialize fal client explicitly
fal.config({
  credentials: process.env.FAL_API_KEY,
});

console.log('FAL client configured with credentials');

// ===== Utility Functions =====
const utils = {
  sleep: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
  
  async imageUrlToBase64(imageUrl) {
    try {
      const response = await fetch(imageUrl);
      const inputBuffer = await response.buffer();
      // Normalize to JPEG so the declared media_type always matches the bytes.
      // Bluesky serves images as webp/png/etc., which otherwise mismatch the
      // hardcoded image/jpeg media type and get rejected by the API.
      const jpegBuffer = await sharp(inputBuffer).jpeg({ quality: 85 }).toBuffer();
      return jpegBuffer.toString('base64');
    } catch (error) {
      console.error('Error converting image to base64:', error);
      return null;
    }
  },

  truncateResponse(text, maxLength = 300) {
    if (text.length <= maxLength) return text;
    
    const searchEnd = Math.min(maxLength, text.length);
    const searchStart = Math.max(0, searchEnd - 50);
    const segment = text.substring(searchStart, searchEnd);
    
    const lastSentenceEnd = Math.max(
      segment.lastIndexOf('.'),
      segment.lastIndexOf('?'),
      segment.lastIndexOf('!')
    );
    
    if (lastSentenceEnd !== -1) {
      return text.substring(0, searchStart + lastSentenceEnd + 1).trim();
    }
    
    const lastSpace = text.lastIndexOf(' ', maxLength - 3);
    return lastSpace !== -1 
      ? text.substring(0, lastSpace) + '...'
      : text.substring(0, maxLength - 3) + '...';
  }
};

// ===== Rate Limiting =====
const RateLimit = {
  limits: {
    hourlyRequests: 0,
    dailyRequests: 0,
    lastHourReset: Date.now(),
    lastDayReset: Date.now()
  },

  check() {
    const now = Date.now();
    
    if (now - this.limits.lastHourReset > 3600000) {
      this.limits.hourlyRequests = 0;
      this.limits.lastHourReset = now;
    }
    
    if (now - this.limits.lastDayReset > 86400000) {
      this.limits.dailyRequests = 0;
      this.limits.lastDayReset = now;
    }
    
    this.limits.hourlyRequests++;
    this.limits.dailyRequests++;
    
    if (this.limits.hourlyRequests > 100 || this.limits.dailyRequests > 1000) {
      throw new Error('Rate limit exceeded');
    }
  }
};

// Replace static ALLOWED_USERS with a Set for better performance
let ALLOWED_USERS = new Set();

// Create a base Bot class
class BaseBot {
  constructor(config, agent) {
    this.config = config;
    this.agent = agent;
    this.repliedPosts = new Set();
    this.did = null; // set on authenticate()
  }

  async generateResponse(post, context) {
    throw new Error('generateResponse must be implemented by child class');
  }

  async generateImagePrompt(post, response) {
    throw new Error('generateImagePrompt must be implemented by child class');
  }

  // Shared methods can go here
  async hasAlreadyReplied(post) {
    try {
      // First check our local Set
      if (this.repliedPosts.has(post.uri)) {
        console.log('Found post in local reply history');
        return true;
      }

      console.log(`Checking for existing replies to post: ${post.uri}`);
      const { data: thread } = await this.agent.getPostThread({
        uri: post.uri,
        depth: 1,
        parentHeight: 1
      });

      console.log(`Thread data:`, JSON.stringify(thread, null, 2));

      if (thread.thread.replies && thread.thread.replies.length > 0) {
        const hasReply = thread.thread.replies.some(reply => 
          reply.post.author.handle === this.config.BLUESKY_IDENTIFIER
        );
        if (hasReply) {
          // Add to our Set if we find an existing reply
          this.repliedPosts.add(post.uri);
        }
        console.log(`Has existing reply: ${hasReply}`);
        return hasReply;
      }
      console.log(`No replies found`);
      return false;
    } catch (error) {
      console.error('Error checking for existing replies:', error);
      return false;
    }
  }

  async monitor() {
    let consecutiveErrors = 0;
    const MAX_RETRIES = 5;
    const BACKOFF_DELAY = 60000; // 1 minute

    try {
      await this.authenticate();
      
      console.log('Starting monitoring...');

      while (true) {
        try {
          const posts = await this.getRecentPosts();

          if (!posts.length) {
            console.log('No posts found');
            await utils.sleep(this.config.CHECK_INTERVAL);
            continue;
          }

          // Handle the most recent notifications, oldest-first so replies go out
          // in chronological order. hasAlreadyReplied + repliedPosts make this
          // safe to run every cycle without duplicate responses.
          const batch = posts.slice(0, this.config.MAX_PER_CHECK).reverse();
          for (const item of batch) {
            try {
              await this.handleNotification(item);
            } catch (err) {
              console.error('Error handling notification:', err);
              this.repliedPosts.add(item.post.uri); // don't retry a poison post
            }
          }

          consecutiveErrors = 0; // Reset error counter on success
          await utils.sleep(this.config.CHECK_INTERVAL);

        } catch (error) {
          console.error('Error in monitoring loop:', error);
          consecutiveErrors++;
          
          if (consecutiveErrors >= MAX_RETRIES) {
            console.error(`Maximum retries (${MAX_RETRIES}) reached, restarting monitor...`);
            break;
          }
          
          const delay = BACKOFF_DELAY * Math.pow(2, consecutiveErrors - 1);
          console.log(`Retrying in ${delay/1000} seconds...`);
          await utils.sleep(delay);
        }
      }
    } catch (error) {
      console.error('Fatal error in monitor:', error);
      await utils.sleep(BACKOFF_DELAY);
    }
  }

  // Decide whether to reply to a single notification, and do so if appropriate
  async handleNotification(item) {
    const mentionsBot = item.post?.record?.text?.includes(this.config.BLUESKY_IDENTIFIER);

    // Only answer a reply if it's a *direct* reply to one of our own posts
    // (parent authored by us). Bluesky also notifies us about deeper replies
    // within our threads, but we don't want to jump into those conversations.
    const isDirectReplyToBot =
      item.reason === 'reply' &&
      !!this.did &&
      item.post?.record?.reply?.parent?.uri?.startsWith(`at://${this.did}/`);

    if (!mentionsBot && !isDirectReplyToBot) return;

    if (await this.hasAlreadyReplied(item.post)) return;

    const context = await this.getReplyContext(item.post);

    // Loop guard: stop once we've already posted enough times in this thread,
    // so the bot can't get stuck ping-ponging with other bots.
    const ownReplies = context.filter(
      msg => msg.author === this.config.BLUESKY_IDENTIFIER
    ).length;

    if (ownReplies >= this.config.MAX_THREAD_REPLIES) {
      console.log(`Already replied ${ownReplies} times in this thread (max ${this.config.MAX_THREAD_REPLIES}), skipping to avoid loops`);
      this.repliedPosts.add(item.post.uri);
      return;
    }

    // For direct replies (not explicit mentions), ask Haiku whether this is
    // actually something the bot should weigh in on — replies to the bot are
    // sometimes just tagging another person, side chatter, etc.
    if (isDirectReplyToBot && !mentionsBot) {
      const warrants = await this.shouldReply(item.post, context);
      if (!warrants) {
        console.log('Reply judged not to warrant a response, skipping');
        this.repliedPosts.add(item.post.uri);
        return;
      }
    }

    console.log('Generating and posting response...');
    const response = await this.generateResponse(item.post, context);

    if (response) {
      await this.postReply(item.post, response);
      // Add rate limiting delay after posting
      await utils.sleep(2000);
    }
  }

  // Ask Haiku whether a direct reply actually warrants a response from the bot
  async shouldReply(post, context) {
    try {
      const conversation = context
        .map(msg => `${msg.author}: ${msg.text}`)
        .join('\n');

      const judgment = await this.anthropic.messages.create({
        model: 'claude-haiku-4-5',
        max_tokens: 5,
        messages: [{
          role: 'user',
          content: `You are a filter for a Bluesky bot with the handle "${this.config.BLUESKY_IDENTIFIER}". Someone has replied to one of the bot's posts. Decide whether this reply actually warrants a response from the bot — i.e. it is genuinely addressed to the bot and invites or expects a reply. If the reply is not really directed at the bot (for example it just tags another person, is a side conversation between others, or otherwise doesn't call for a response), it does NOT warrant a reply.\n\nConversation so far (oldest first):\n${conversation}\n\nMost recent reply:\n${post.author.handle}: ${post.record.text}\n\nShould the bot reply? Answer with only "yes" or "no".`
        }]
      });

      const answer = (judgment.content[0]?.text || '').trim().toLowerCase();
      const yes = answer.startsWith('y');
      console.log(`shouldReply judgment: "${answer}" -> ${yes}`);
      return yes;
    } catch (error) {
      console.error('Error in shouldReply check:', error);
      return true; // fail open: if the check errors, fall back to replying
    }
  }

  // Function to authenticate with Bluesky
  async authenticate() {
    try {
      const { data } = await this.agent.login({
        identifier: this.config.BLUESKY_IDENTIFIER,
        password: this.config.BLUESKY_APP_PASSWORD,
      });
      this.did = data.did; // our own DID, used to detect direct replies to us
      console.log(`Successfully authenticated with Bluesky as ${this.did}`);
    } catch (error) {
      console.error('Authentication failed:', error);
      throw error;
    }
  }

  // Function to get recent posts from the monitored handle
  async getRecentPosts() {
    try {
      const { data: notifications } = await this.agent.listNotifications({
        limit: 20
      });
      
      if (!notifications || !notifications.notifications) {
        throw new Error('No notifications returned');
      }
      
      // Respond to mentions and to replies to our own posts, excluding self
      const relevantPosts = notifications.notifications
        .filter(notif => {
          // Only mentions and replies (not likes, reposts, follows, etc.)
          if (notif.reason !== 'mention' && notif.reason !== 'reply') return false;

          // Prevent self-replies
          return notif.author.handle !== this.config.BLUESKY_IDENTIFIER;
        })
        .map(notif => ({
          reason: notif.reason,
          post: {
            uri: notif.uri,
            cid: notif.cid,
            author: notif.author,
            record: notif.record
          }
        }));

      console.log(`Found ${relevantPosts.length} relevant notifications`);
      return relevantPosts;
    } catch (error) {
      console.error('Error in getRecentPosts:', error);
      return [];
    }
  }

  // Function to get the context of a reply
  async getReplyContext(post) {
    try {
      const conversation = [];
      
      // Helper function to extract images
      const extractImages = (record) => {
        const images = record?.embed?.images || 
                      record?.embed?.media?.images || 
                      [];
        return images.map(img => ({
          alt: img.alt,
          url: img.fullsize || img.thumb
        }));
      };

      // Add the main post
      conversation.push({
        author: post.author.handle,
        text: post.record.text,
        images: extractImages(post.record)
      });

      if (post.record.embed?.$type === 'app.bsky.embed.record' && post.record.embed) {
        try {
          console.log('Processing embed record');
          const uri = post.record.embed.record.uri;
          const matches = uri.match(/at:\/\/([^/]+)\/[^/]+\/([^/]+)/);
          
          if (matches) {
            const [_, repo, rkey] = matches;
            
            // Fetch the full quoted post data
            const quotedPostResponse = await this.agent.getPost({
              repo,
              rkey
            });

            if (quotedPostResponse?.value) {
              // Extract author from the repo DID
              const authorDid = matches[1];
              const postValue = quotedPostResponse.value;
              
              // Only add to conversation if we have the text
              if (postValue.text) {
                console.log('Found a quote post with text:', postValue.text);
                
                // Extract images if present
                const quotedImages = postValue.embed?.images || 
                                   postValue.embed?.media?.images || 
                                   [];
                
                conversation.unshift({
                  author: authorDid, // We might want to convert this DID to a handle
                  text: postValue.text,
                  images: quotedImages.map(img => ({
                    alt: img.alt,
                    url: img.fullsize || img.thumb
                  }))
                });
              }
            }
          }
        } catch (error) {
          console.error('Error fetching quoted post:', error);
        }
      }

      // Then get the thread history if it's a reply
      if (post.record?.reply) {
        let currentUri = post.uri;

        while (currentUri) {
          const { data: thread } = await this.agent.getPostThread({
            uri: currentUri,
            depth: 0,
            parentHeight: 1
          });

          if (!thread.thread.post) break;

          // Extract image information if present
          const images = thread.thread.post.record.embed?.images || 
                        thread.thread.post.record.embed?.media?.images || 
                        [];
          
          conversation.unshift({
            author: thread.thread.post.author.handle,
            text: thread.thread.post.record.text,
            images: images.map(img => ({
              alt: img.alt,
              url: thread.thread.post.embed?.images?.[0]?.fullsize || 
                   thread.thread.post.embed?.images?.[0]?.thumb ||
                   img.image?.fullsize || 
                   img.image?.thumb
            }))
          });

          if (thread.thread.parent && thread.thread.parent.post) {
            currentUri = thread.thread.parent.post.uri;
          } else {
            break;
          }
        }
      }

      console.log('Conversation context with images:', JSON.stringify(conversation, null, 2));
      return conversation;
    } catch (error) {
      console.error('Error fetching reply context:', error);
      return [];
    }
  }

  // Modify the postReply function
  async postReply(post, response) {
    try {
      RateLimit.check();
      
      // Generate image prompt using Claude Haiku
      const imagePrompt = await this.generateImagePrompt(post, response);
      console.log('Generated image prompt:', imagePrompt);

      // Generate image using Fal
      const falResult = await fal.subscribe("fal-ai/flux-2/klein/9b", {
        input: {
          prompt: imagePrompt,
          num_inference_steps: 4,
          image_size: {
            width: 768,
            height: 768
          },
          num_images: 1,
          enable_safety_checker: true,
          output_format: "png"
        },
        logs: true
      });
      
      console.log('FAL.ai response:', falResult);
      const imageUrl = falResult.data.images[0].url;

      // Fetch the image directly
      const imageResponse = await fetch(imageUrl);
      const imageBuffer = await imageResponse.buffer();

      // Convert to JPEG, preserving the generated aspect ratio. Bound the
      // longest side so the blob stays under Bluesky's ~1MB image limit.
      const processedImageBuffer = await sharp(imageBuffer)
        .resize(1024, 1024, {
          fit: 'inside',
          withoutEnlargement: true
        })
        .jpeg({ quality: 80 })
        .toBuffer();

      // Upload directly to Bluesky
      const uploadResult = await this.agent.uploadBlob(processedImageBuffer, {
        encoding: 'image/jpeg'
      });

      // Create detailed alt text
      const altText = [
        `Response Model: ${this.getModelName()}`,
        'Response Prompt: Text and images upthread of this comment',
        'Image Prompt Model: Claude Haiku 4.5',
        `Image Prompt: ${imagePrompt}`,
        'Image Generation Model: Fal AI FLUX.2 [klein] 9B'
      ].join('\n');

      const replyObject = {
        text: utils.truncateResponse(response),
        embed: {
          $type: 'app.bsky.embed.images',
          images: [{
            alt: altText,
            image: uploadResult.data.blob
          }]
        },
        reply: {
          root: post.record?.reply?.root || { uri: post.uri, cid: post.cid },
          parent: { uri: post.uri, cid: post.cid }
        }
      };

      const result = await this.agent.post(replyObject);
      console.log('Successfully posted reply:', result.uri);
      this.repliedPosts.add(post.uri);
    } catch (error) {
      console.error('Error posting reply:', error);
      this.repliedPosts.add(post.uri); // Prevent retries
    }
  }

  // Add a method to get model name
  getModelName() {
    return 'Unknown Model';
  }
}

// Claude-specific implementation
class ClaudeBot extends BaseBot {
  constructor(config, agent) {
    super(config, agent);
    this.anthropic = new Anthropic({
      apiKey: config.CLAUDE_API_KEY,
    });
  }

  async generateResponse(post, context) {
    try {
      let messages = [{ 
        role: 'user', 
        content: [
          {
            type: 'text',
            text: `You are part of a bot designed to respond to a conversation on Bluesky. You will write a reply, and another part of the bot will post it. Here's the context, with the oldest message first:\n\n`
          }
        ]
      }];

      // Add context messages with images
      if (context && context.length > 0) {
        for (const msg of context) {
          let content = [{
            type: 'text',
            text: `${msg.author}: ${msg.text}`
          }];
          
          // Add images if present
          if (msg.images && msg.images.length > 0) {
            console.log(`Processing ${msg.images.length} images for message`);
            for (const image of msg.images) {
              console.log(`Converting image to base64: ${image.url}`);
              const base64Image = await utils.imageUrlToBase64(image.url);
              if (base64Image) {
                content.push({
                  type: 'image',
                  source: {
                    type: 'base64',
                    media_type: 'image/jpeg', // Adjust based on actual image type if needed
                    data: base64Image
                  }
                });
                if (image.alt) {
                  content.push({
                    type: 'text',
                    text: `[Image description: ${image.alt}]`
                  });
                }
              }
            }
          }
          
          messages[0].content.push(...content);
        }
      }
      
      console.log('Final messages object:', JSON.stringify(messages, null, 2));
      
      // Add the final message
      messages[0].content.push({
        type: 'text',
        text: `\nThe most recent message mentioning you is: "${post.record.text}"\n\nPlease respond to the request in the most recent message in 300 characters or less. Your response will be posted to BlueSky as a reply to the most recent message mentioning you by a bot. `
      });

      const message = await this.anthropic.messages.create({
        model: 'claude-opus-4-8',
        max_tokens: 150,
        messages: messages
      });

      return message.content[0].text;
    } catch (error) {
      console.error('Error generating Claude response:', error);
      return null;
    }
  }

  async generateImagePrompt(post, response) {
    const promptMessage = await this.anthropic.messages.create({
      model: 'claude-haiku-4-5',
      max_tokens: 150,
      messages: [{
        role: 'user',
        content: `Create a prompt for an image model based on the following question and answer. If the prompt doesn't already have animals in it, add cats.\n\nQ: ${post.record.text}\nA: ${response}`
      }]
    });
    return promptMessage.content[0].text;
  }

  getModelName() {
    return 'Claude Opus 4.8';
  }
}

// DeepSeek implementation
class DeepSeekBot extends BaseBot {
  constructor(config, agent) {
    super(config, agent);
    this.deepseek = new OpenAI({
      apiKey: config.DEEPSEEK_API_KEY,
      baseURL: "https://api.deepseek.com"
    });
    this.anthropic = new Anthropic({
      apiKey: config.CLAUDE_API_KEY,
    });
  }

  async generateResponse(post, context) {
    try {
      // Add debug logging
      console.log('Initializing DeepSeek API call...');
      
      // Format context into a conversation history
      let conversationHistory = '';
      if (context && context.length > 0) {
        for (const msg of context) {
          conversationHistory += `${msg.author}: ${msg.text}\n`;
          if (msg.images && msg.images.length > 0) {
            msg.images.forEach(image => {
              if (image.alt) {
                conversationHistory += `[Image description: ${image.alt}]\n`;
              }
            });
          }
        }
      }

      // Log the request we're about to make
      console.log('Making DeepSeek API request with config:', {
        model: "deepseek-chat",
        baseURL: this.deepseek.baseURL,
        hasApiKey: !!this.config.DEEPSEEK_API_KEY
      });

      const response = await this.deepseek.chat.completions.create({
        model: "deepseek-chat",
        messages: [
          {
            role: "system",
            content: "You are part of a bot designed to respond to a conversation on Bluesky. You will write a reply, and another part of the bot will post it. Keep your responses under 300 characters."
          },
          {
            role: "user",
            content: `Here's the conversation context:\n\n${conversationHistory}\nThe most recent message mentioning you is: "${post.record.text}"\n\nPlease respond to the request in the most recent message.`
          }
        ],
        temperature: 0.7,
        max_tokens: 150
      });

      // Log successful response
      console.log('DeepSeek API response received successfully');
      
      return response.choices[0].message.content;
    } catch (error) {
      // Enhanced error logging
      console.error('Detailed DeepSeek API error:', {
        message: error.message,
        status: error.status,
        response: error.response,
        stack: error.stack
      });

      // If it's an API error with a response
      if (error.response) {
        try {
          const errorBody = await error.response.text();
          console.error('API Error Response:', errorBody);
        } catch (e) {
          console.error('Could not parse error response');
        }
      }

      // Fallback response for production
      return "I apologize, but I'm having trouble connecting to my reasoning service right now. Please try again later.";
    }
  }

  // Use Claude Haiku for image prompts, same as ClaudeBot
  async generateImagePrompt(post, response) {
    const promptMessage = await this.anthropic.messages.create({
      model: 'claude-haiku-4-5',
      max_tokens: 150,
      messages: [{
        role: 'user',
        content: `Create a prompt for an image model based on the following question and answer. If the prompt doesn't already have animals in it, add either whales or cats, it's your choice!.\n\nQ: ${post.record.text}\nA: ${response}`
      }]
    });
    return promptMessage.content[0].text;
  }

  getModelName() {
    return 'DeepSeek v3';
  }
}

// Initialize and run multiple bots
async function startBots() {
  const agent1 = new AtpAgent({ service: 'https://bsky.social' });
  const agent2 = new AtpAgent({ service: 'https://bsky.social' });

  const claudeBot = new ClaudeBot({
    ...config,
    BLUESKY_IDENTIFIER: config.CLAUDE_IDENTIFIER,
    BLUESKY_APP_PASSWORD: config.CLAUDE_APP_PASSWORD,
  }, agent1);

  const deepseekBot = new DeepSeekBot({
    ...config,
    BLUESKY_IDENTIFIER: config.DEEPSEEK_IDENTIFIER,
    BLUESKY_APP_PASSWORD: config.DEEPSEEK_APP_PASSWORD,
  }, agent2);

  // Run bots in parallel
  await Promise.all([
    claudeBot.monitor().catch(console.error),
    deepseekBot.monitor().catch(console.error)
  ]);
}

// Start the bots
startBots().catch(console.error);