/**
 * CL-GRU4Rec+RP Demo - Main Page
 * E-commerce style demo with real-time recommendations from FastAPI backend
 */

"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { apiClient, Product } from "@/lib/api-client";
import { ProductCard } from "@/components/ProductCard";
import { RecommendationPanel } from "@/components/RecommendationPanel";
import { SessionTracker } from "@/components/SessionTracker";
import { LogOut, ShoppingCart, AlertCircle } from "lucide-react";

export default function HomePage() {
  const { user, logout, isAuthenticated } = useAuth();
  const [sessionIds, setSessionIds] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<any>(null);

  // Real data state
  const [products, setProducts] = useState<Product[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState<string | null>(null);
  const [productMap, setProductMap] = useState<Map<string, Product>>(new Map());

  // Load products and categories on mount
  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setApiError(null);

        // Check API health
        const health = await apiClient.healthCheck();

        if (!health.model_loaded || !health.products_loaded) {
          console.warn("API not fully loaded:", health);
        }

        // Load categories first
        const cats = await apiClient.getCategories();
        setCategories(cats);

        // Load products (limit to 100 initially for performance)
        const productsData = await apiClient.getProducts(undefined, 100);
        setProducts(productsData.products);

        // Build product map for quick lookup
        const map = new Map<string, Product>();
        productsData.products.forEach((p) => map.set(p.id, p));
        setProductMap(map);
      } catch (error) {
        console.error("Failed to load data:", error);
        setApiError(
          "Không thể kết nối đến API. Vui lòng đảm bảo FastAPI backend đang chạy.",
        );
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  // Update recommendations when session changes
  useEffect(() => {
    if (sessionIds.length === 0) {
      setRecommendations(null);
      return;
    }

    async function fetchRecommendations() {
      try {
        const result = await apiClient.getRecommendations(
          sessionIds,
          6,
          user?.id,
        );
        setRecommendations(result);
      } catch (error) {
        console.error("Failed to get recommendations:", error);
      }
    }

    fetchRecommendations();
  }, [sessionIds, user]);

  const handleProductClick = (productId: string) => {
    // Add to session if not already present
    if (!sessionIds.includes(productId)) {
      setSessionIds([...sessionIds, productId]);
    }
  };

  const handleRemoveFromSession = (productId: string) => {
    setSessionIds(sessionIds.filter((id) => id !== productId));
  };

  const handleClearSession = () => {
    setSessionIds([]);
    setRecommendations(null);
  };

  const filteredProducts = selectedCategory
    ? products.filter((p) => p.main_category === selectedCategory)
    : products;

  const sessionProducts = sessionIds
    .map((id) => productMap.get(id))
    .filter(Boolean) as Product[];

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Đang tải dữ liệu...</p>
        </div>
      </div>
    );
  }

  // API Error state
  if (apiError) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center gap-3 text-red-600 mb-4">
            <AlertCircle size={24} />
            <h2 className="text-lg font-semibold">Lỗi kết nối API</h2>
          </div>
          <p className="text-gray-600 mb-4">{apiError}</p>
          <div className="bg-gray-50 p-4 rounded-lg text-sm">
            <p className="font-semibold mb-2">Khắc phục:</p>
            <ol className="list-decimal list-inside space-y-1 text-gray-600">
              <li>
                Chạy FastAPI backend:{" "}
                <code className="bg-gray-200 px-1 rounded">
                  cd backend && python -m app.main
                </code>
              </li>
              <li>Đảm bảo Python 3.14+ và dependencies đã cài đặt</li>
              <li>Kiểm tra xem port 8000 có đang được sử dụng không</li>
            </ol>
          </div>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center p-4">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">Vui lòng đăng nhập</h1>
          <a href="/login" className="text-blue-600 underline">
            Đăng nhập
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShoppingCart className="text-blue-600" />
            <h1 className="text-xl font-bold">CL-GRU4Rec+RP Demo</h1>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">
              Xin chào, <span className="font-semibold">{user?.name}</span>
            </span>
            <button
              onClick={logout}
              className="flex items-center gap-1 text-sm text-gray-600 hover:text-red-600"
            >
              <LogOut size={16} />
              Đăng xuất
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Sidebar - Session & Filters */}
          <div className="space-y-4">
            <SessionTracker
              products={sessionProducts}
              onRemove={handleRemoveFromSession}
              onClear={handleClearSession}
            />

            {/* Category Filter */}
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="font-semibold text-gray-900 mb-3">📁 Danh mục</h3>
              <div className="space-y-1 max-h-64 overflow-y-auto">
                <button
                  onClick={() => setSelectedCategory(null)}
                  className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
                    selectedCategory === null
                      ? "bg-blue-600 text-white"
                      : "hover:bg-gray-100 text-gray-700"
                  }`}
                >
                  Tất cả ({products.length})
                </button>
                {categories.map((cat) => {
                  const count = products.filter(
                    (p) => p.main_category === cat,
                  ).length;
                  return (
                    <button
                      key={cat}
                      onClick={() => setSelectedCategory(cat)}
                      className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
                        selectedCategory === cat
                          ? "bg-blue-600 text-white"
                          : "hover:bg-gray-100 text-gray-700"
                      }`}
                    >
                      {cat} ({count})
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Demo Info */}
            <div className="bg-blue-50 rounded-lg p-4 text-sm">
              <h3 className="font-semibold text-blue-900 mb-2">ℹ️ Hướng dẫn</h3>
              <ol className="space-y-1 text-blue-800 list-decimal list-inside">
                <li>Chọn sản phẩm để thêm vào session</li>
                <li>Watch gợi ý cập nhật real-time</li>
                <li>Xem điểm components (GRU, CL, RP)</li>
                <li>Click vào gợi ý để thêm tiếp</li>
              </ol>
            </div>
          </div>

          {/* Center - Product Grid */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-4 mb-4">
              <h2 className="font-semibold text-gray-900">
                {selectedCategory || "Tất cả sản phẩm"}
              </h2>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {filteredProducts.map((product) => (
                <ProductCard
                  key={product.id}
                  product={product}
                  onClick={() => handleProductClick(product.id)}
                />
              ))}
            </div>
          </div>

          {/* Right - Recommendations */}
          <div>
            <RecommendationPanel
              result={recommendations}
              products={productMap}
              onProductClick={handleProductClick}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-8 py-4">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-600">
          <p>
            CL-GRU4Rec+RP Demo - Contrasting Learning Enhanced GRU4Rec with
            Re-Purchase Awareness
          </p>
        </div>
      </footer>
    </div>
  );
}
