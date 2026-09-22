const JSON_HEADERS = { "Content-Type": "application/json" };

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const type = response.headers.get("content-type") || "";
  const payload = type.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : response.statusText;
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

export const api = {
  get: (url) => request(url),
  post: (url, body) => request(url, { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body ?? {}) }),
  put: (url, body) => request(url, { method: "PUT", headers: JSON_HEADERS, body: JSON.stringify(body ?? {}) }),
  del: (url) => request(url, { method: "DELETE" }),
  upload: (url, file, field = "file") => {
    const form = new FormData();
    form.append(field, file);
    return request(url, { method: "POST", body: form });
  },
};

export const projectApi = (id) => ({
  base: `/api/projects/${id}`,
  detail: () => api.get(`/api/projects/${id}`),
  samples: (params = {}) => api.get(`/api/projects/${id}/samples?${new URLSearchParams(params)}`),
  boxes: (name) => api.get(`/api/projects/${id}/boxes/${encodeURIComponent(name)}`),
  saveLabels: (name, body) => api.post(`/api/projects/${id}/labels/${encodeURIComponent(name)}`, body),
  imageUrl: (name) => `/api/projects/${id}/image/${encodeURIComponent(name)}`,
  selection: (body) => api.post(`/api/projects/${id}/selection`, body),
  selections: () => api.get(`/api/projects/${id}/selections`),
  selectionDetail: (file) => api.get(`/api/projects/${id}/selections/${file}`),
  embeddings: (force = false) => api.post(`/api/projects/${id}/embeddings?force=${force}`),
  heatmap: (clusters) => api.get(`/api/projects/${id}/heatmap?clusters=${clusters}`),
  buildHeatmap: (clusters, force = false) =>
    api.post(`/api/projects/${id}/heatmap?clusters=${clusters}&force=${force}`),
  stats: () => api.get(`/api/projects/${id}/stats`),
  buildStats: (params) => api.post(`/api/projects/${id}/stats?${new URLSearchParams(params)}`),
  trainConfig: () => api.get(`/api/projects/${id}/train-config`),
  saveTrainConfig: (config) => api.put(`/api/projects/${id}/train-config`, { config }),
  trainBundle: (config) => api.post(`/api/projects/${id}/train-bundle`, { config }),
  models: () => api.get(`/api/projects/${id}/models`),
  linkModel: (path) => api.post(`/api/projects/${id}/models/link`, { path }),
  uploadModel: (file) => api.upload(`/api/projects/${id}/models/upload`, file),
  deleteModel: (name) => api.del(`/api/projects/${id}/models/${encodeURIComponent(name)}`),
  tune: (body) => api.post(`/api/projects/${id}/tune`, body),
  wbf: () => api.get(`/api/projects/${id}/wbf`),
  inferBundle: (includeWeights) =>
    api.post(`/api/projects/${id}/infer-bundle`, { include_weights: includeWeights }),
  uploadSubmission: (file) => api.upload(`/api/projects/${id}/submission/upload`, file),
  linkSubmission: (path) => api.post(`/api/projects/${id}/submission/path`, { path }),
  dropSubmission: () => api.del(`/api/projects/${id}/submission`),

  /* --------------------------------------------------- проверка разметки */
  quality: (limit = 400) => api.get(`/api/projects/${id}/quality?limit=${limit}`),
  buildQuality: (body) => api.post(`/api/projects/${id}/quality`, body),

  /* ------------------------------------------------------ статус кадров */
  status: () => api.get(`/api/projects/${id}/status`),
  flag: (name, body) => api.post(`/api/projects/${id}/flag/${encodeURIComponent(name)}`, body),
  neighbors: (name, k = 12) =>
    api.get(`/api/projects/${id}/neighbors/${encodeURIComponent(name)}?k=${k}`),
  saveLabelsBatch: (body) => api.post(`/api/projects/${id}/labels-batch`, body),

  /* --------------------------------------------------------- train / val */
  planSplit: (body) => api.post(`/api/projects/${id}/split/plan`, body),
  applySplit: (body) => api.post(`/api/projects/${id}/split/apply`, body),
  exportDataset: (body) => api.post(`/api/projects/${id}/export/dataset`, body),

  /* ------------------------------------------------------------ задания */
  candidates: (body) => api.post(`/api/projects/${id}/candidates`, body),
  assignments: () => api.get(`/api/projects/${id}/assignments`),
  createAssignment: (body) => api.post(`/api/projects/${id}/assignments`, body),
  assignment: (aid) => api.get(`/api/projects/${id}/assignments/${aid}`),
  cancelAssignment: (aid) => api.del(`/api/projects/${id}/assignments/${aid}`),
  exportAssignment: (aid, mode) =>
    api.post(`/api/projects/${id}/assignments/${aid}/export`, { mode }),
  assignmentResult: (aid) => api.post(`/api/projects/${id}/assignments/${aid}/result`, {}),
  acceptAssignment: (path) => api.post(`/api/projects/${id}/assignments/accept`, { path }),
  importResult: (path, assignment = null) =>
    api.post(`/api/projects/${id}/assignments/import`, { path, assignment }),

  /* -------------------------------------------------------- предразметка */
  prelabel: (body) => api.post(`/api/projects/${id}/prelabel`, body),
  proposals: () => api.get(`/api/projects/${id}/proposals`),
  applyProposals: (body) => api.post(`/api/projects/${id}/proposals/apply`, body),
  dropProposals: () => api.del(`/api/projects/${id}/proposals`),
});

export const uploadApi = {
  /** Кладёт файл во временную папку и возвращает путь — для drag & drop. */
  stage: (file) => api.upload("/api/projects/uploads", file),
};

export const fsApi = {
  roots: () => api.get("/api/fs/roots"),
  list: (path, showFiles = false) =>
    api.get(`/api/fs/list?path=${encodeURIComponent(path)}&show_files=${showFiles}`),
  inspect: (path) => api.get(`/api/fs/inspect?path=${encodeURIComponent(path)}`),
  search: (root, pattern) =>
    api.get(`/api/fs/search?root=${encodeURIComponent(root)}&pattern=${encodeURIComponent(pattern)}`),
};

export const taskApi = {
  get: (taskId) => api.get(`/api/tasks/${taskId}`),
};
