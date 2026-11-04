"use client";

import React, { KeyboardEvent } from "react";

export interface TabItem<T extends string = string> {
  id: T;
  label: string;
  icon?: React.ReactNode;
  badge?: string | number;
  badgeColor?: string;
  disabled?: boolean;
  content?: React.ReactNode;
}

export interface GenericTabsProps<T extends string = string> {
  tabs: TabItem<T>[];
  activeTab: T;
  onChange: (tabId: T) => void;
  variant?: "pills" | "underline" | "segmented";
  size?: "sm" | "md" | "lg";
  className?: string;
  tabListClassName?: string;
  contentClassName?: string;
  rightAction?: React.ReactNode;
  renderContent?: boolean;
}

export function GenericTabs<T extends string = string>({
  tabs,
  activeTab,
  onChange,
  variant = "pills",
  size = "md",
  className = "",
  tabListClassName = "",
  contentClassName = "",
  rightAction,
  renderContent = false,
}: GenericTabsProps<T>) {
  const activeTabItem = tabs.find((t) => t.id === activeTab);

  // Keyboard navigation support for accessible tabs
  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const enabledTabs = tabs.filter((t) => !t.disabled);
    const currentIndex = enabledTabs.findIndex((t) => t.id === activeTab);
    if (currentIndex === -1) return;

    if (e.key === "ArrowRight") {
      e.preventDefault();
      const nextTab = enabledTabs[(currentIndex + 1) % enabledTabs.length];
      if (nextTab) onChange(nextTab.id);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      const prevTab = enabledTabs[(currentIndex - 1 + enabledTabs.length) % enabledTabs.length];
      if (prevTab) onChange(prevTab.id);
    }
  };

  // Base styling per size
  const sizeClasses = {
    sm: "px-2.5 py-1 text-xs",
    md: "px-3 py-1.5 text-xs",
    lg: "px-4 py-2 text-sm",
  }[size];

  // Base styling per variant
  const getTabClasses = (tab: TabItem<T>, isActive: boolean) => {
    if (tab.disabled) {
      return "opacity-40 cursor-not-allowed text-slate-400 bg-slate-50";
    }

    if (variant === "underline") {
      return `border-b-2 font-bold transition-all flex items-center gap-2 ${sizeClasses} ${
        isActive
          ? "border-blue-600 text-blue-700 bg-blue-50/50"
          : "border-transparent text-slate-500 hover:text-slate-900 hover:border-slate-300"
      }`;
    }

    if (variant === "segmented") {
      return `rounded-md transition-all font-semibold flex items-center gap-1.5 ${sizeClasses} ${
        isActive
          ? "bg-blue-600 text-white font-bold shadow-xs"
          : "text-slate-600 hover:text-slate-900 hover:bg-white/60"
      }`;
    }

    // Default: "pills"
    return `rounded-lg font-mono font-bold uppercase transition-all flex items-center gap-2 border ${sizeClasses} ${
      isActive
        ? "bg-blue-50 border-blue-600 text-blue-800 shadow-xs"
        : "bg-slate-100 text-slate-600 border-slate-200 hover:text-slate-900 hover:bg-slate-200"
    }`;
  };

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Tab Navigation Header Strip */}
      <div className={`flex flex-wrap items-center justify-between gap-3 ${variant === "underline" ? "border-b border-slate-200" : ""}`}>
        <div
          role="tablist"
          onKeyDown={handleKeyDown}
          className={`flex items-center gap-2 overflow-x-auto ${
            variant === "segmented"
              ? "bg-slate-100 p-1 rounded-lg border border-slate-200"
              : ""
          } ${tabListClassName}`}
        >
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                id={`tab-${tab.id}`}
                aria-selected={isActive}
                aria-controls={`panel-${tab.id}`}
                tabIndex={isActive ? 0 : -1}
                disabled={tab.disabled}
                onClick={() => onChange(tab.id)}
                className={getTabClasses(tab, isActive)}
              >
                {tab.icon && <span className="shrink-0">{tab.icon}</span>}
                <span>{tab.label}</span>
                {tab.badge !== undefined && (
                  <span
                    className={`ml-1 px-1.5 py-0.2 rounded text-[10px] font-sans font-bold leading-tight ${
                      tab.badgeColor || (isActive ? "bg-blue-200 text-blue-900" : "bg-slate-200 text-slate-700")
                    }`}
                  >
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {rightAction && <div className="shrink-0">{rightAction}</div>}
      </div>

      {/* Render active tab content automatically if renderContent is true */}
      {renderContent && activeTabItem?.content && (
        <div
          role="tabpanel"
          id={`panel-${activeTabItem.id}`}
          aria-labelledby={`tab-${activeTabItem.id}`}
          className={`transition-all ${contentClassName}`}
        >
          {activeTabItem.content}
        </div>
      )}
    </div>
  );
}

export default GenericTabs;
