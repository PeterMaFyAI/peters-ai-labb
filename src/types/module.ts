export type ResourceType = "visualization" | "lab";
export type ModuleStatus = "planned" | "ready";

export interface LearningModule {
  id: string;
  title: string;
  slug: string;
  description: string;
  imageUrl?: string;
  category: string;
  targetGroup: string;
  route: string;
  type: ResourceType;
  status: ModuleStatus;
  keywords: string[];
}
