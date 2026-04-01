/**
 * Product Card Component
 * Displays product with click handler
 */

"use client";

import { Product } from "@/lib/mock-data";

interface ProductCardProps {
  product: Product;
  onClick?: () => void;
  variant?: "default" | "compact";
}

export function ProductCard({ product, onClick, variant = "default" }: ProductCardProps) {
  if (variant === "compact") {
    return (
      <button
        onClick={onClick}
        className="w-full text-left p-2 rounded hover:bg-gray-100 transition-colors flex items-center gap-2"
      >
        <span className="text-2xl">{product.image}</span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{product.name}</p>
          <p className="text-xs text-gray-500">{product.category}</p>
        </div>
        <p className="text-sm font-semibold text-blue-600">
          {product.price.toLocaleString()}đ
        </p>
      </button>
    );
  }

  return (
    <button
      onClick={onClick}
      className="w-full bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow p-4 border border-gray-200 hover:border-blue-300"
    >
      <div className="aspect-square bg-gray-50 rounded-lg flex items-center justify-center mb-3">
        <span className="text-6xl">{product.image}</span>
      </div>
      <h3 className="font-semibold text-gray-900 mb-1 line-clamp-2">{product.name}</h3>
      <p className="text-sm text-gray-500 mb-2">{product.category}</p>
      <p className="text-lg font-bold text-blue-600">
        {product.price.toLocaleString()}đ
      </p>
    </button>
  );
}
