// Shared API types between server and plugin

export interface SearchRequest {
  query: string
  user_id: string
  limit?: number
}

export interface SearchResult {
  id: string
  memory: string
  score: number
  user_id?: string
  agent_id?: string
}

export interface SearchResponse {
  results: SearchResult[]
}

export interface AddRequest {
  messages: string
  user_id: string
  agent_id?: string
}

export interface AddResult {
  id: string
  memory: string
  event: string
}

export interface AddResponse {
  results: AddResult[]
}

export interface DeleteResponse {
  success: boolean
}

export interface HealthResponse {
  status: string
}

export interface Memory {
  id: string
  memory: string
  user_id?: string
  agent_id?: string
  created_at?: string
  updated_at?: string
}
