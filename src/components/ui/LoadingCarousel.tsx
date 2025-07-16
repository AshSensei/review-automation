// components/LoadingCarousel.tsx
import { motion, AnimatePresence } from 'framer-motion';

interface StatusUpdate {
  message: string;
  timestamp: number;
}

interface LoadingCarouselProps {
  // It now accepts a single StatusUpdate object, or null
  status: StatusUpdate | null;
}

// A standard utility class for screen-reader-only content
const srOnlyStyle: React.CSSProperties = {
  position: 'absolute',
  width: '1px',
  height: '1px',
  padding: '0',
  margin: '-1px',
  overflow: 'hidden',
  clip: 'rect(0, 0, 0, 0)',
  whiteSpace: 'nowrap',
  border: '0',
};

export const LoadingCarousel = ({ status }: LoadingCarouselProps) => {
  return (
    <section 
      aria-label="Loading status"
      className="flex flex-col items-center justify-center text-center text-gray-500 h-6" // Give a fixed height to prevent layout shift
    >
      <p style={srOnlyStyle}>Analysis in progress...</p>
      
      <AnimatePresence mode="wait">
        {status && (
          <motion.p
            // The key is now the message itself, which triggers the animation
            key={status.message}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.8, ease: "easeInOut" }}
            aria-live="polite"
            aria-atomic="true"
          >
            {status.message}
          </motion.p>
        )}
      </AnimatePresence>
    </section>
  );
};