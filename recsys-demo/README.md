# CL-GRU4Rec+RP Web Demo

Interactive web demo for CL-GRU4Rec+RP recommendation system.

## Tech Stack

- **Next.js 16** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS v4** - Styling
- **Mock Auth** - localStorage-based authentication

## Features

- 🔐 **Mock Authentication**: Sign in with any email/password
- 🛒 **Product Catalog**: 15 products across 5 categories
- 🎯 **Real-time Recommendations**: Updates as you add products to session
- 📊 **Component Scores**: See GRU4Rec, Contrastive Learning, and Re-Purchase scores
- 💡 **Session Tracking**: Visualize current session and remove items
- 📁 **Category Filtering**: Filter products by category

## Getting Started

```bash
# Run development server
npm run dev --turbo

# Open browser
# http://localhost:3000
```

## Demo Credentials

Any email/password works! Quick test:
- Email: `demo@example.com`
- Password: `demo123`

Or use pre-configured:
- Email: `user@demo.com`
- Password: any

## How to Use

1. **Login**: Enter any email/password
2. **Browse Products**: View catalog on the left
3. **Add to Session**: Click any product to add to your session
4. **View Recommendations**: Watch real-time recommendations on the right
5. **See Component Scores**: Each recommendation shows:
   - **GRU**: Sequential pattern score
   - **CL**: Contrastive learning similarity
   - **RP**: Re-purchase frequency
6. **Continue Shopping**: Click recommended products to extend session
7. **Remove Items**: Use X button to remove from session

## File Structure

```
src/
├── app/
│   ├── page.tsx              # Main demo page
│   ├── login/page.tsx        # Login page
│   └── layout.tsx            # Root layout with AuthProvider
├── components/
│   ├── ProductCard.tsx       # Product display
│   ├── RecommendationPanel.tsx  # Recommendations with scores
│   └── SessionTracker.tsx    # Current session display
├── contexts/
│   └── AuthContext.tsx       # Mock authentication
├── lib/
│   ├── mock-data.ts          # Product catalog
│   └── recommendation-engine.ts  # CL-GRU4Rec+RP simulation
```

## Component Architecture

### Recommendation Engine

The demo simulates CL-GRU4Rec+RP with three components:

1. **GRU4Rec (Sequential)**
   - Scores based on category co-occurrence
   - Position-based boost for recent items
   - Confidence increases with session length

2. **Contrastive Learning (Similarity)**
   - Simulates embedding similarity
   - Products with similar IDs get higher scores
   - Independent of session length

3. **Re-Purchase (History)**
   - Frequency-based scoring within session
   - Normalized by session length

### Adaptive Fusion

Weights adapt based on session length:
- Short session (< 3 items): GRU=50%, CL=25%, RP=25%
- Medium session (3-7 items): GRU=65%, CL=15%, RP=20%
- Long session (> 7 items): GRU=80%, CL=10%, RP=10%

## Notes

- This is a **demo/simulation** - not the actual trained model
- Recommendation scores are simulated for demonstration
- Real CL-GRU4Rec+RP model requires PyTorch training
- Use this for presentations, demos, and interactive explanations

## Future Enhancements

- Connect to real PyTorch model via API
- Add more products and categories
- Implement actual embedding similarity
- Add user history persistence
- Include evaluation metrics dashboard
