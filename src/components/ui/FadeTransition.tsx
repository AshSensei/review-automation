// components/FadeTransition.tsx
import { motion, AnimatePresence } from 'framer-motion';
import React from 'react';

interface FadeTransitionProps {
  children: React.ReactNode;
  animationKey: string | number;
}

export const FadeTransition = ({ children, animationKey }: FadeTransitionProps) => {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={animationKey}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.5, ease: "easeInOut" }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
};