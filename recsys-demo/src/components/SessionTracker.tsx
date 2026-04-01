/**
 * Session Tracker Component
 * Displays current session products
 */

"use client";

import { Product } from "@/lib/api-client";
import { X } from "lucide-react";

interface SessionTrackerProps {
  products: Product[];
  onRemove?: (productId: string) => void;
  onClear?: () => void;
}

// Get emoji icon based on category
function getCategoryIcon(category: string): string {
  const categoryIcons: Record<string, string> = {
    "Коконы для новорожденных": "👶",
    Электрокачели: "🎠",
    "Качели, шезлонги": "🪑",
    Коляски: "👼",
    Манеж: "🏠",
    Ходунки: "🚶",
    "Стульчики для кормления": "🍼",
    default: "📦",
  };
  return categoryIcons[category] || categoryIcons.default;
}

export function SessionTracker({
  products,
  onRemove,
  onClear,
}: SessionTrackerProps) {
  if (products.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-gray-900">🛒 Session hiện tại</h3>
          <span className="text-xs text-gray-500">0 sản phẩm</span>
        </div>
        <p className="text-gray-400 text-sm text-center py-4">
          Chưa có sản phẩm nào
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-900">🛒 Session hiện tại</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {products.length} sản phẩm
          </span>
          {onClear && (
            <button
              onClick={onClear}
              className="text-xs text-red-600 hover:text-red-700 underline"
            >
              Xóa tất cả
            </button>
          )}
        </div>
      </div>

      <div className="space-y-2 max-h-64 overflow-y-auto">
        {products.map((product, idx) => (
          <div
            key={product.id}
            className="flex items-center gap-2 p-2 bg-gray-50 rounded group"
          >
            <span className="text-xs text-gray-400 w-4">{idx + 1}</span>
            <span className="text-xl">
              {getCategoryIcon(product.main_category)}
            </span>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate text-gray-900">
                {product.name}
              </p>
              <p className="text-xs text-gray-500">{product.main_category}</p>
            </div>
            {onRemove && (
              <button
                onClick={() => onRemove(product.id)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-200 rounded transition-opacity"
              >
                <X size={16} className="text-gray-500" />
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Session info */}
      <div className="mt-3 pt-3 border-t text-xs text-gray-500">
        <p>
          💡 Session length:{" "}
          <span className="font-semibold">{products.length}</span>
        </p>
        <p className="text-gray-400 mt-1">
          {products.length < 3
            ? "Session ngắn - Gợi ý dựa trên popularity"
            : products.length < 7
              ? "Session trung bình - Kết hợp GRU + CL"
              : "Session dài - GRU4Rec chiếm ưu thế"}
        </p>
      </div>
    </div>
  );
}
