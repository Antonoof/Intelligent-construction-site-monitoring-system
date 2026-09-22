/**
 * Cluster labels people can actually remember.
 *
 * "k17" tells you nothing and reads the same as "k18"; a name sticks after one
 * glance. The mapping is deterministic by cluster id, so the same group keeps its
 * name across the map, the annotation queue and the dashboard.
 */
const NAMES = [
  "Крош", "Ёжик", "Нюша", "Бараш", "Копатыч", "Лосяш", "Совунья", "Пин",
  "Кар-Карыч", "Биби", "Пыжик", "Шуршик", "Плюшик", "Хрумчик", "Топтыш", "Свистун",
  "Бубенчик", "Мигунчик", "Чихун", "Пузырик", "Клюквик", "Ворчун", "Мурлыка", "Скрипун",
  "Пончик", "Ушастик", "Топтыжка", "Дрёмыч", "Пискун", "Хвостик", "Лужик", "Фонарик",
  "Тучкин", "Веснушка", "Кисточка", "Гвоздик", "Шнурок", "Компотик", "Бурчун", "Зевака",
  "Сопелка", "Кудряш", "Кувырок", "Барабанчик", "Ласточкин", "Пыхтун", "Умняш", "Егоза",
];

export function clusterName(id) {
  const index = Math.abs(Number(id) || 0);
  const base = NAMES[index % NAMES.length];
  const round = Math.floor(index / NAMES.length);
  return round ? `${base} ${round + 1}` : base;
}

/** Name plus the raw id, for tooltips and tables where both matter. */
export const clusterLabel = (id) => `${clusterName(id)} · k${id}`;
