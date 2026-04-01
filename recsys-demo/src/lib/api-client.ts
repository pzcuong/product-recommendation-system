/**
 * API Client for CL-GRU4Rec+RP Backend
 * Connects to FastAPI backend with real model
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Product {
  id: string;
  name: string;
  brand: string;
  main_category: string;
  categories: string;
  price: number;
  description: string;
  slug: string;
}

export interface Recommendation {
  product_id: string;
  score: number;
  gru_score: number;
  cl_score: number;
  rp_score: number;
}

export interface RecommendationResult {
  recommendations: Recommendation[];
  confidence: number;
  session_length: number;
  component_weights: {
    gru: number;
    cl: number;
    rp: number;
  };
}

export interface ProductsResponse {
  products: Product[];
  total: number;
  categories: string[];
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options?.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  async getProducts(category?: string, limit?: number): Promise<ProductsResponse> {
    const params = new URLSearchParams();
    if (category) params.append("category", category);
    if (limit) params.append("limit", limit.toString());

    const endpoint = `/api/products${params.toString() ? `?${params}` : ""}`;
    return this.request<ProductsResponse>(endpoint);
  }

  async getProduct(productId: string): Promise<Product> {
    return this.request<Product>(`/api/products/${productId}`);
  }

  async getCategories(): Promise<string[]> {
    const response = await this.request<{ categories: string[] }>("/api/categories");
    return response.categories;
  }

  async getRecommendations(
    sessionItems: string[],
    k: number = 10,
    clientId?: string
  ): Promise<RecommendationResult> {
    return this.request<RecommendationResult>("/api/recommend", {
      method: "POST",
      body: JSON.stringify({
        session_items: sessionItems,
        k,
        client_id: clientId,
      }),
    });
  }

  async getPopularProducts(limit: number = 50): Promise<{ products: Product[] }> {
    return this.request<{ products: Product[] }>(`/api/popular?limit=${limit}`);
  }

  async healthCheck(): Promise<{ status: string; model_loaded: boolean; products_loaded: boolean }> {
    return this.request<{ status: string; model_loaded: boolean; products_loaded: boolean }>("/health");
  }
}

export const apiClient = new ApiClient();
