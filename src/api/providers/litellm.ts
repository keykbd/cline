import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"
import { ApiHandlerOptions, liteLlmDefaultModelId, liteLlmModelInfoSaneDefaults } from "../../shared/api"
import { ApiHandler } from ".."
import { ApiStream } from "../transform/stream"
import { convertToOpenAiMessages } from "../transform/openai-format"

export class LiteLlmHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.client = new OpenAI({
			baseURL: this.options.liteLlmBaseUrl || "http://localhost:4000",
			apiKey: this.options.liteLlmApiKey || "noop",
		})
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const formattedMessages = convertToOpenAiMessages(messages)
		const systemMessage: OpenAI.Chat.ChatCompletionSystemMessageParam = {
			role: "system",
			content: systemPrompt,
		}
		const modelId = this.options.liteLlmModelId || liteLlmDefaultModelId

		// Check if extended thinking is enabled
		const budget_tokens = this.options.thinkingBudgetTokens || 0
		const reasoningOn = budget_tokens > 0

		// Prepare request parameters
		const requestParams: OpenAI.ChatCompletionCreateParams = {
			model: modelId,
			messages: [systemMessage, ...formattedMessages],
			// Temperature is not set when using extended thinking
			// Claude documentation states that temperature is not compatible with thinking
			temperature: reasoningOn ? undefined : 0,
			stream: true,
			stream_options: { include_usage: true },
		}

		// Add extended thinking parameters if enabled
		if (reasoningOn) {
			// Add thinking parameter for Claude models via LiteLLM
			// This assumes LiteLLM properly passes this parameter to Claude models
			;(requestParams as any).thinking = {
				type: "enabled",
				budget_tokens: budget_tokens,
			}
		}

		const stream = await this.client.chat.completions.create(requestParams)

		for await (const chunk of stream) {
			// Handle standard text content
			const delta = chunk.choices[0]?.delta
			if (delta?.content) {
				yield {
					type: "text",
					text: delta.content,
				}
			}

			// Handle reasoning/thinking content if present
			// This assumes LiteLLM formats thinking content in a similar way to Anthropic's API
			if ((chunk as any).delta?.thinking) {
				yield {
					type: "reasoning",
					reasoning: (chunk as any).delta.thinking,
				}
			} else if ((chunk as any).content_block?.type === "thinking") {
				yield {
					type: "reasoning",
					reasoning: (chunk as any).content_block.thinking || "",
				}
			}

			// Handle usage statistics
			if (chunk.usage) {
				yield {
					type: "usage",
					inputTokens: chunk.usage.prompt_tokens || 0,
					outputTokens: chunk.usage.completion_tokens || 0,
				}
			}
		}
	}

	getModel() {
		return {
			id: this.options.liteLlmModelId || liteLlmDefaultModelId,
			info: liteLlmModelInfoSaneDefaults,
		}
	}
}
