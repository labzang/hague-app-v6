import axios from 'axios'

// FastAPI 백엔드(app/api_server.py)와 통신하는 기본 URL
const API_URL = 'http://localhost:8000' // 하드코딩으로 고정
console.log('API_URL:', API_URL)
console.log('Environment NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL)

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10초 타임아웃
})

// 요청 인터셉터 추가
api.interceptors.request.use(
  (config) => {
    console.log('Request config:', config)
    return config
  },
  (error) => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// 응답 인터셉터 추가
api.interceptors.response.use(
  (response) => {
    console.log('Response:', response)
    return response
  },
  (error) => {
    console.error('Response error:', error)
    return Promise.reject(error)
  }
)

export interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: Date
  sources?: DocumentSource[]
}

export interface DocumentSource {
  content: string
  metadata?: Record<string, any>
}

export interface RAGResponse {
  question: string
  answer: string
  retrieved_documents: Array<{
    content: string
    metadata?: Record<string, any>
  }>
  retrieved_count: number
}

export const chatAPI = {
  async sendMessage(question: string, k: number = 3): Promise<RAGResponse> {
    const requestData = { question, k }
    console.log('Sending LangChain request to:', `${API_URL}/api/products`)
    console.log('Request body:', requestData)

    try {
      // LangChain RAG 엔드포인트 호출
      const response = await fetch(`${API_URL}/api/products`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      })

      console.log('Response status:', response.status)
      console.log('Response headers:', response.headers)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Error response:', errorText)
        throw new Error(`HTTP ${response.status}: ${errorText}`)
      }

      const data = await response.json()
      console.log('Response data:', data)
      return data
    } catch (error) {
      console.error('Fetch error:', error)
      throw error
    }
  },

  async sendGraphMessage(question: string, k: number = 3): Promise<RAGResponse> {
    const requestData = { question, k }
    console.log('Sending LangGraph request to:', `${API_URL}/api/graph`)
    console.log('Request body:', requestData)

    try {
      // LangGraph 엔드포인트 호출
      const response = await fetch(`${API_URL}/api/graph`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      })

      console.log('Response status:', response.status)
      console.log('Response headers:', response.headers)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Error response:', errorText)
        throw new Error(`HTTP ${response.status}: ${errorText}`)
      }

      const data = await response.json()
      console.log('Response data:', data)
      return data
    } catch (error) {
      console.error('Fetch error:', error)
      throw error
    }
  },

  async healthCheck(): Promise<{ status: string }> {
    const response = await fetch(`${API_URL}/health`)
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    return response.json()
  },

  async recommendProducts(message: string, data?: Record<string, any>): Promise<any> {
    const requestData = {
      message: message,
      ...data
    }
    console.log('Sending product recommendation request to:', `${API_URL}/api/v1/admin/products/recommend`)
    console.log('Request body:', requestData)

    try {
      const response = await fetch(`${API_URL}/api/v1/admin/products/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      })

      console.log('Response status:', response.status)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Error response:', errorText)
        throw new Error(`HTTP ${response.status}: ${errorText}`)
      }

      const result = await response.json()
      console.log('Response data:', result)
      return result
    } catch (error) {
      console.error('Fetch error:', error)
      throw error
    }
  },
}

