#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import express from 'express';
import fs from 'fs/promises';
import path from 'path';

const app = express();
app.use(express.json());

// Create MCP server
const server = new Server({
    name: 'filesystem-server',
    version: '0.1.0',
}, {
    capabilities: {
        tools: {},
    },
});

// Helper function to validate file paths
function validatePath(filePath) {
    const allowedDirs = ['/mcp-data/data', '/mcp-data/scratch_pad', '/mcp-data/memory'];
    const resolvedPath = path.resolve(filePath);
    return allowedDirs.some(dir => resolvedPath.startsWith(path.resolve(dir)));
}

// Register tools using the correct API
server.setRequestHandler('tools/list', async () => {
    return {
        tools: [
            {
                name: 'read_file',
                description: 'Read the contents of a file',
                inputSchema: {
                    type: 'object',
                    properties: {
                        path: {
                            type: 'string',
                            description: 'Path to the file to read',
                        },
                    },
                    required: ['path'],
                },
            },
            {
                name: 'write_file',
                description: 'Write content to a file',
                inputSchema: {
                    type: 'object',
                    properties: {
                        path: {
                            type: 'string',
                            description: 'Path to the file to write',
                        },
                        content: {
                            type: 'string',
                            description: 'Content to write to the file',
                        },
                    },
                    required: ['path', 'content'],
                },
            },
            {
                name: 'list_directory',
                description: 'List contents of a directory',
                inputSchema: {
                    type: 'object',
                    properties: {
                        path: {
                            type: 'string',
                            description: 'Path to the directory to list',
                        },
                    },
                    required: ['path'],
                },
            },
        ],
    };
});

server.setRequestHandler('tools/call', async (request) => {
    const { name, arguments: args } = request.params;

    switch (name) {
        case 'read_file':
            if (!validatePath(args.path)) {
                throw new Error('Access denied: Path not in allowed directories');
            }
            try {
                const content = await fs.readFile(args.path, 'utf-8');
                return {
                    content: [
                        {
                            type: 'text',
                            text: content,
                        },
                    ],
                };
            } catch (error) {
                throw new Error(`Failed to read file: ${error.message}`);
            }

        case 'write_file':
            if (!validatePath(args.path)) {
                throw new Error('Access denied: Path not in allowed directories');
            }
            try {
                // Ensure directory exists
                await fs.mkdir(path.dirname(args.path), { recursive: true });
                await fs.writeFile(args.path, args.content, 'utf-8');
                return {
                    content: [
                        {
                            type: 'text',
                            text: `Successfully wrote to ${args.path}`,
                        },
                    ],
                };
            } catch (error) {
                throw new Error(`Failed to write file: ${error.message}`);
            }

        case 'list_directory':
            if (!validatePath(args.path)) {
                throw new Error('Access denied: Path not in allowed directories');
            }
            try {
                const entries = await fs.readdir(args.path, { withFileTypes: true });
                const items = entries.map(entry => ({
                    name: entry.name,
                    type: entry.isDirectory() ? 'directory' : 'file',
                }));

                return {
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify(items, null, 2),
                        },
                    ],
                };
            } catch (error) {
                throw new Error(`Failed to list directory: ${error.message}`);
            }

        default:
            throw new Error(`Unknown tool: ${name}`);
    }
});

// SSE endpoint for MCP communication
app.get('/sse', (req, res) => {
    console.log('SSE connection request received');

    try {
        // Create SSE transport
        const transport = new SSEServerTransport('/sse', res);

        // Connect the server to the transport
        server.connect(transport).catch(error => {
            console.error('Failed to connect server to transport:', error);
        });

        // Handle client disconnect
        req.on('close', () => {
            console.log('SSE client disconnected');
        });

        req.on('error', (error) => {
            console.error('SSE request error:', error);
        });

    } catch (error) {
        console.error('Error setting up SSE:', error);
        res.status(500).send('Internal Server Error');
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Start server
const PORT = process.env.FILESYSTEM_SERVER_PORT || 3001;
app.listen(PORT, () => {
    console.log(`Filesystem MCP server listening on port ${PORT}`);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('Shutting down MCP server...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('Shutting down MCP server...');
    process.exit(0);
});
