"use client";

import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import NProgress from "nprogress";

export default function ProgressLink({ href, children, ...props }) {
  const router = useRouter();
  const pathname = usePathname();

  const handleClick = (e) => {
    e.preventDefault();

    // Check if we're already on this page
    const isSamePage = pathname === href;

    if (isSamePage) {
      // Just show a quick flash and complete immediately
      NProgress.start();
      setTimeout(() => {
        NProgress.done();
      }, 200);
    } else {
      // Normal navigation
      NProgress.start();
      setTimeout(() => {
        router.push(href);
      }, 400);
    }
  };

  return (
    <Link href={href} onClick={handleClick} {...props}>
      {children}
    </Link>
  );
}
