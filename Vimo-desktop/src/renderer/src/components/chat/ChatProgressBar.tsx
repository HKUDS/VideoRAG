import React, { useId, useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { AnalysisStep } from '../../types/chat';

interface ChatProgressBarProps {
  content: string;
  analysisSteps: AnalysisStep[];
}

export const ChatProgressBar: React.FC<ChatProgressBarProps> = ({ content, analysisSteps }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const stepsId = useId();

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <div className="p-4">
      <button
        type="button"
        className="flex w-full items-center justify-between mb-3 text-left cursor-pointer hover:bg-gray-50 rounded-lg p-2 -m-2 transition-colors"
        onClick={toggleCollapse}
        aria-expanded={!isCollapsed}
        aria-controls={stepsId}
      >
        <span className="flex items-center">
          <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
          <span className="text-sm font-medium text-gray-900">{content}</span>
        </span>
        <span className="flex items-center">
          {isCollapsed ? (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronUp className="w-4 h-4 text-gray-500" />
          )}
        </span>
      </button>
      
      <div id={stepsId} className="space-y-2 relative" hidden={isCollapsed}>
        {analysisSteps.map((step) => (
          <div key={step.id} className="flex items-center space-x-3 relative">
            {/* Status Indicator */}
            <div className="flex-shrink-0">
              {step.status === 'completed' ? (
                <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
              ) : step.status === 'error' ? (
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              ) : (
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              )}
            </div>

            {/* Step Content */}
            <div className="flex-1 min-w-0">
              <div className={`text-sm ${
                step.status === 'completed'
                  ? 'text-gray-500'
                  : step.status === 'error'
                    ? 'text-red-600'
                    : 'text-gray-900'
              }`}>
                <span className="font-medium">{step.name}</span>
                <span className="ml-2 font-normal text-xs">{step.message}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}; 