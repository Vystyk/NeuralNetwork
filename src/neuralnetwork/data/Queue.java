package neuralnetwork.data;

import java.util.List;

public class Queue<T> {
	
	T value;
	Element first;
	Element last;
	int size = 0;
	int max = 100;
	
	public Queue() {};
	
	public Queue(int max) {
		setMaxSize(max);
	}
	
	public void addElement(T e){
		if(size >= max){
			Element element = new Element(e);
			first.previous = element;
			element.next = first;
			first = element;
			last = last.previous;
			last.next = null;
		}else if(size == 0){
			first = new Element(e);
			last = first;
			size++;
		} else {
			Element element = new Element(e);
			first.previous = element;
			element.next = first;
			first = element;
			size++;
		}
		
		
	}
	
	public float average(){
		float total = 0;
		Element current = first;
		while(current != null){
			total += (double)current.element;
			current = current.next;
		}
		
		return total / size;
	}
	
	public void setMaxSize(int size){ this.size = size; }
	
	class Element{
		
		public T element;
		public Element next;
		public Element previous;
		
		public Element(T e){ element = e; }
		
	}
}
