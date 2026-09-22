import { api } from "../core/api.js";
import { qs, qsa } from "../core/dom.js";
import { confirmDialog, notifyError, toast } from "../core/ui.js";
import { pickFolder } from "../components/file-picker.js";

qs("#open-project")?.addEventListener("click", async () => {
  const path = await pickFolder("Выберите папку проекта (с файлом project.json)");
  if (!path) return;

  try {
    const { project } = await api.post("/api/projects/open", { path });
    toast("Проект открыт", { kind: "success" });
    window.location.href = `/p/${project.id}`;
  } catch (error) {
    notifyError(error);
  }
});

qs("#project-filter")?.addEventListener("input", (event) => {
  const needle = event.target.value.trim().toLowerCase();
  qsa(".project-card").forEach((card) => {
    card.classList.toggle("hidden", needle && !card.dataset.name.includes(needle));
  });
});

qsa("[data-delete]").forEach((button) => {
  button.addEventListener("click", async () => {
    const id = button.dataset.delete;
    const ok = await confirmDialog(
      `Проект «${id}» будет удалён вместе с кэшем, выборками и экспортами. Исходные изображения не тронуты.`,
      { title: "Удалить проект?" }
    );
    if (!ok) return;

    try {
      await api.del(`/api/projects/${id}`);
      button.closest(".project-card").remove();
      toast("Проект удалён", { kind: "success" });
    } catch (error) {
      notifyError(error);
    }
  });
});
