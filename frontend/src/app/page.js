"use client";

import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";

export default function Home() {
  const router = useRouter();
  const [isExiting, setIsExiting] = useState(false);
  const [overlayVisible, setOverlayVisible] = useState(false);

  const handlePlayClick = () => {
    setIsExiting(true);
    setOverlayVisible(true);

    setTimeout(() => {
      setOverlayVisible(false);
    }, 3000);

    setTimeout(() => {
      router.push("/home");
    }, 4000);
  };

  return (
    <AnimatePresence mode="wait">
      {!isExiting && (
        <motion.div
          key="landing"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          className="relative w-full h-screen flex flex-col justify-center px-64 bg-white"
        >
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2, ease: "easeOut" }}
            className="text-7xl font-bold bg-gradient-to-r from-black via-primary-dark to-primary bg-clip-text text-transparent leading-snug"
          >
            A Sentiment Analysis Framework for Assessing Filipino Public
            Perception of Climate Change
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 1 }}
            className="flex flex-row gap-4 text-xl py-4"
          >
            <p className="font-semibold text-gray-mid">BARTOLOME</p>
            <p className="font-semibold text-gray-mid">•</p>
            <p className="font-semibold text-gray-mid">BUMANGLAG</p>
            <p className="font-semibold text-gray-mid">•</p>
            <p className="font-semibold text-gray-mid">ROLLE</p>
          </motion.div>

          <motion.button
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.8, duration: 0.8 }}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className="bg-primary text-2xl text-white font-semibold tracking-wider py-4 w-1/6 rounded-full mt-8 shadow-xl hover:bg-primary-dark active:bg-gray-dark transition-all duration-300"
            onClick={handlePlayClick}
          >
            Start
          </motion.button>
        </motion.div>
      )}

      {isExiting && (
        <AnimatePresence>
          {overlayVisible && (
            <motion.div
              key="exit-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 1 }}
              className="absolute inset-0 flex flex-col items-center justify-center bg-white z-50 backdrop-blur-2xl"
            >
              <div className="relative w-32 h-32">
                <motion.div
                  className="absolute inset-0 rounded-full border-[6px] border-t-transparent border-b-transparent border-l-primary border-r-blue blur-[1px]"
                  animate={{ rotate: 360 }}
                  transition={{
                    repeat: Infinity,
                    duration: 1.8,
                    ease: "linear",
                  }}
                />
                <div className="absolute inset-5 rounded-full bg-gradient-to-br from-white to-bluish-gray shadow-inner" />
                <motion.div
                  className="absolute inset-0 rounded-full bg-gradient-to-tr from-primary/30 via-transparent to-transparent blur-lg"
                  animate={{ rotate: -360 }}
                  transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                />
              </div>

              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 1 }}
                className="text-gray-dark text-lg font-medium mt-8"
              >
                Preparing...
              </motion.p>
            </motion.div>
          )}
        </AnimatePresence>
      )}
    </AnimatePresence>
  );
}
